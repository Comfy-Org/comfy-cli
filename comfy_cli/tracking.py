from __future__ import annotations

import atexit
import copy
import functools
import json
import logging as logginglib
import os
import queue
import re
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol

import typer

from comfy_cli import constants, logging, ui
from comfy_cli.caller import detect_caller, stream_is_tty
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager

if TYPE_CHECKING:
    from posthog import Posthog

# Ignore logs from urllib3 that Mixpanel/PostHog use.
logginglib.getLogger("urllib3").setLevel(logginglib.ERROR)

# posthog-python reports every failed upload at ERROR on the "posthog" logger
# (it never logs above ERROR). Telemetry is best-effort by contract — a failed
# upload must not look like a product failure on the user's stderr — so silence
# it entirely unless the user is explicitly debugging (LOG_LEVEL=DEBUG).
if os.environ.get("LOG_LEVEL", "").upper() != "DEBUG":
    logginglib.getLogger("posthog").setLevel(logginglib.CRITICAL)

MIXPANEL_TOKEN = "93aeab8962b622d431ac19800ccc9f67"

# phc_* are public client-side write keys designed for embedding — safe to commit, same as MIXPANEL_TOKEN above.
# Override with $POSTHOG_API_KEY (see _resolve_posthog_token for the accepted shapes).
_POSTHOG_DEFAULT_TOKEN = "phc_iKfK86id4xVYws9LybMje0h44eGtfwFgRPIBehmy8rO"
POSTHOG_HOST = "https://t.comfy.org"


def _resolve_posthog_token() -> str:
    """Resolve the PostHog project write key, guarding against the
    $POSTHOG_API_KEY name collision: PostHog's own tooling uses that name for a
    personal (phx_) API key, which the ingestion endpoint rejects with a 401.
    Only a phc_* project write key is accepted as an override; an empty string
    still disables the provider (existing escape hatch); anything else is
    ignored in favor of the committed default.
    """
    raw = os.environ.get("POSTHOG_API_KEY")
    if raw is None:
        return _POSTHOG_DEFAULT_TOKEN
    if raw == "" or raw.startswith("phc_"):
        return raw
    # ERROR, not WARNING: the CLI's default LOG_LEVEL is ERROR (see
    # logging.setup_logging), so a WARNING here would be silently dropped and
    # the user would never learn their override was ignored.
    logging.error(
        "Ignoring $POSTHOG_API_KEY: not a phc_* project write key "
        "(personal phx_ keys cannot ingest events); using the built-in key."
    )
    return _POSTHOG_DEFAULT_TOKEN


# Only these events get the tracing_id --> workflow_run_id alias on PostHog.
EXECUTION_EVENTS = frozenset({"execution_start", "execution_success", "execution_error"})

# Namespace applied to event names on PostHog only, matching the
# app:/hub:/registry: surface-prefix convention in the shared project. Mixpanel
# keeps the bare legacy names (see ``mixpanel_name`` in track_event) so its
# historical streams stay continuous.
POSTHOG_EVENT_PREFIX = "cli:"

# Sanitize command kwargs before sending them as telemetry: _is_sensitive()
# masks credential-bearing names, _is_trackable() drops ctx/private/unserializable
# values, and _scrub_value() strips query strings off URL values.

_SENSITIVE_SUFFIXES = ("_token", "_api_key", "_secret", "_password")
# `token` is the publish PAT; `changelog` is bulky free text with no analytics
# value beyond its presence. `key` is the bare `--key` option carrying the Comfy
# Cloud API key (e.g. `cloud set-key`, auth store). `prompt` (the `comfy run
# --prompt` positive prompt) and `set_overrides` (the `--set` field=value list)
# are verbatim user content — like `changelog`, we keep the presence but never
# ship the text. Sensitive values become "<redacted>" (the key is kept so we can
# still tell the option was supplied).
# `from_` is the `--from` path: a local filesystem path naming the user's home
# directory and their install layout, with no analytics value beyond having been
# supplied.
_SENSITIVE_EXACT = frozenset(
    {"api_key", "key", "token", "password", "secret", "changelog", "prompt", "set_overrides", "from_"}
)


def _is_sensitive(name: str) -> bool:
    """True if *name* looks like a credential. Case-insensitive; matches the
    snake_case suffixes only (Typer kwargs are always snake_case)."""
    lower = name.lower()
    return lower in _SENSITIVE_EXACT or lower.endswith(_SENSITIVE_SUFFIXES)


def _is_trackable(name: str, value: object) -> bool:
    """True if the (name, value) kwarg is safe to send. Drops ctx/context,
    underscore-prefixed names, and values json can't serialize -- posthog-python
    coerces unserializable values and ships them (e.g. a Click Context) rather
    than raising the way Mixpanel does, so we must reject them ourselves."""
    if name in ("ctx", "context"):
        return False
    if name.startswith("_"):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


# Any `<scheme>://` prefix, not just http(s): a credential can ride the userinfo
# slot of ftp/ssh/redis/etc. just as easily, and the scrub below is safe for all
# of them (it only ever removes the query, fragment, and userinfo components).
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def _scrub_value(value: object) -> object:
    """Strip the credential-bearing components out of a top-level URL string.

    Two of them, both observed in the wild:

    - **query string and fragment** — CivitAI download links carry the API
      token as ``?token=``.
    - **userinfo** — ``scheme://user:secret@host/path`` puts a basic-auth
      password in the authority. Reachable through ``COMFY_USER_AGENT``, whose
      value becomes the ``caller_kind`` stamped on every event, so a harness
      that self-attributes with a service URL would otherwise ship its own
      credential to both providers on every command.

    Non-URL strings are returned verbatim. That leaves a bare filesystem path
    (or a ``file://`` one) carrying a username untouched — deliberately: there
    is no general way to tell a private path segment from a public one, and
    such a value is not a credential. Redacting the whole value is the job of
    ``_is_sensitive``, keyed on the property name.
    """
    if not isinstance(value, str) or not _URL_SCHEME_RE.match(value):
        return value
    scheme, sep, rest = value.partition("://")
    # ORDER MATTERS: userinfo comes off FIRST, and on the last `@` in the whole
    # remainder — before any path/query/fragment split.
    #
    # Splitting the query off first (or bounding the authority at the first
    # `/`) assumes a well-formed URL, and the values that worry us are exactly
    # the malformed ones. `https://svc:s3cr?et@host/agent` would strand the
    # first half of the password in the result, and `https://svc:ab/cd@host/p`
    # would return verbatim with the password intact — base64-ish tokens
    # routinely contain `/` and `?`. Cutting at the last `@` first makes the
    # credential's own contents irrelevant to where the cut lands.
    #
    # This errs toward removing too much: a legal-but-unusual path containing
    # an unencoded `@` loses its leading segments. That is the correct bias for
    # a scrub — the cost is analytics detail, the alternative is a leak.
    rest = rest.rpartition("@")[2]
    rest = rest.partition("?")[0].partition("#")[0]
    return f"{scheme}{sep}{rest}"


# The four intrinsic caller kinds are short and fixed, but COMFY_USER_AGENT lets
# a caller name itself anything and detect_caller only lowercases it. 64 chars is
# ample for a self-attribution label ("claude-code", "my-harness/1.2").
_CALLER_KIND_MAX_LEN = 64


def _sanitize_caller_kind(kind: str) -> str:
    """Make the caller label safe to ship as a telemetry property.

    ``caller_kind`` rides EVERY event — including ``feedback_submitted``, which
    is dispatched even when passive-telemetry consent is off — and a custom
    ``COMFY_USER_AGENT`` is arbitrary user-supplied text that could hold a path,
    a URL, or unbounded junk. Give it the same treatment command kwargs get:
    strip credentials embedded in a URL value, then cap the length so a
    pathological label can't ship an unbounded string to both providers.
    """
    scrubbed = _scrub_value(kind)
    text = scrubbed if isinstance(scrubbed, str) else str(scrubbed)
    return text[:_CALLER_KIND_MAX_LEN]


# Generate a unique tracing ID per command.
config_manager = ConfigManager()
cli_version = config_manager.get_cli_version()

# tracking all events for a single user
user_id = config_manager.get(constants.CONFIG_KEY_USER_ID)
# tracking all events for a single command
tracing_id = str(uuid.uuid4())
# Who is driving this process: "user" | "pipe" | "agent" | "claude-code" | a
# lowercased custom COMFY_USER_AGENT label. Computed once at import, matching
# the cli_version/tracing_id pattern above, so we don't re-run isatty per event.
_caller = detect_caller()
_caller_kind = _sanitize_caller_kind(_caller.kind)
# Whether that label is self-attributed free text rather than a kind the CLI
# derived itself. Taken from `source_env`, which is authoritative, NOT from
# membership in _INTRINSIC_CALLER_KINDS: `COMFY_USER_AGENT=user` produces the
# string "user" while being exactly the self-attributed case, so a set-membership
# test would let an agentic caller pass itself off as a human.
_caller_kind_is_custom = _caller.source_env == "COMFY_USER_AGENT"


def _intrinsic_caller_kind() -> str:
    """``_caller_kind``, or ``"custom"`` when it is a self-attributed label.

    For the one send path the user has NOT consented to passively — feedback,
    which ships on an explicit user action and is suppressed only by the hard
    env opt-out — a free-text, environment-derived label is more than the
    analytics question needs. "Was this a human or an agent?" is answered just
    as well by the closed set, so a custom ``COMFY_USER_AGENT`` collapses to
    the literal ``"custom"`` there rather than riding along verbatim.
    """
    return "custom" if _caller_kind_is_custom else _caller_kind


workspace_manager = WorkspaceManager()

# Process-scoped opt-in used when running non-interactively before the
# user has ever recorded a consent choice. Captures agentic usage without
# persisting the consent flag, so a later interactive run can still
# prompt the human. The anonymous user_id is persisted separately for
# stable agent identity in analytics.
_session_only_tracking = False


def _telemetry_disabled_by_env() -> bool:
    """Return True if telemetry is suppressed via environment variable.

    Honors the cross-tool ``DO_NOT_TRACK`` convention
    (https://consoledonottrack.com/) and the project-specific
    ``COMFY_NO_TELEMETRY``. Per the spec, any value other than empty or
    ``"0"`` opts out.
    """
    for name in ("DO_NOT_TRACK", "COMFY_NO_TELEMETRY"):
        val = os.environ.get(name, "")
        if val and val != "0":
            return True
    return False


# Click/Typer completion instruction tokens. The ``_*_COMPLETE`` var carries an
# ``instruction_shell`` pair whose instruction is ``complete`` (resolve args) or
# ``source`` (emit the completion script) — neither runs a command.
_COMPLETION_INSTRUCTIONS = frozenset({"complete", "source"})


def _in_shell_completion() -> bool:
    """Return True when the process is resolving shell tab-completion rather
    than running a real command.

    Click/Typer trigger completion by re-invoking the CLI with a
    ``_<PROG_NAME>_COMPLETE`` environment variable set (e.g. ``_COMFY_COMPLETE``
    under fish, bash, and zsh). No command actually runs on that path, so there
    is no telemetry to send — detecting it lets us skip standing up the PostHog
    client (a background thread + network setup), which is wasted work on an
    inert path (GitHub #506). The prog name varies with the invoking entrypoint
    (``comfy`` / ``comfy-cli`` / ``comfycli``) and any user alias, so match the
    ``_..._COMPLETE`` pattern rather than a fixed name.

    The var's value is Click/Typer's completion *instruction* — ``complete_bash``
    / ``source_zsh`` (Typer 8.x style) or ``bash_complete`` (Click 7.x style),
    i.e. an ``instruction_shell`` / ``shell_instruction`` pair. Require a
    recognized instruction token so a stray or empty user-exported
    ``_FOO_COMPLETE`` can't silently suppress telemetry on a real command run.
    Snapshot the keys with ``list(...)`` so a concurrent env mutation on another
    thread can't raise ``RuntimeError: dictionary changed size during iteration``.
    """
    for name in list(os.environ):
        if not (name.startswith("_") and name.endswith("_COMPLETE")):
            continue
        if _COMPLETION_INSTRUCTIONS & set(os.environ.get(name, "").split("_")):
            return True
    return False


def _consent_enabled() -> bool:
    """Whether passive telemetry may be sent right now: no env opt-out AND the
    user has consented (persisted flag) or a session-only opt-in is active.

    This is the full gate. Agent-authored data (e.g. session reviews) rides it,
    so opting out of tracking by any means means nothing is sent."""
    if _telemetry_disabled_by_env():
        return False
    return bool(config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)) or _session_only_tracking


class TelemetryProvider(Protocol):
    enabled: bool

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


def _log_telemetry_debug(message: str) -> None:
    """Best-effort debug log that can never propagate out of telemetry code.

    Used on the Mixpanel worker's failure paths. The worker is a daemon thread
    that can outlive `_flush_all_providers`' deadline and so log *after* the
    stdlib's own `logging.shutdown` atexit hook has closed the handlers; on top
    of that, a raise from inside an `except` handler would kill the worker for
    the rest of the process. Neither is worth a user-visible traceback.
    """
    try:
        logging.debug(message)
    except BaseException:  # noqa: BLE001  # pragma: no cover - defensive
        pass


class MixpanelProvider:
    # A CLI invocation emits ~1-3 events, so this cap is effectively unreachable
    # outside pathological cases. It exists so a wedged worker (blackholed
    # endpoint) can't let an unbounded backlog accumulate — dropping is the
    # correct trade for best-effort telemetry.
    _QUEUE_MAX = 256

    def __init__(self, token: str):
        self.client = None
        if token:
            # Imported here, not at module scope: see `_get_providers`.
            from mixpanel import Consumer as MixpanelConsumer
            from mixpanel import Mixpanel

            # mixpanel-python's default Consumer uses request_timeout=None → an
            # unbounded, synchronous requests.post, so a blackholed telemetry
            # endpoint (accepts TCP, never responds) hangs whichever thread sends.
            # Sends now happen on the worker below rather than
            # the caller's thread, so this bound caps how long ONE send can occupy
            # the queue — and with it, how much of the atexit drain's shared 5s
            # deadline a single in-flight event can consume. retry_limit=1
            # (default is 4 with backoff) keeps a blackholed send to a single ~10s
            # attempt instead of ~40s+ across retries.
            self.client = Mixpanel(token, consumer=MixpanelConsumer(request_timeout=10, retry_limit=1))
            # Dispatch is queue-and-drain so track() never blocks the caller:
            # @track_command fires its event *before* running the
            # wrapped command body, and `run` fires execution_start before
            # submitting the workflow, so an inline send put a synchronous HTTP
            # round-trip on the hot path of every consented invocation.
            self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAX)
            # flush() waits on this counter rather than `queue.join()`:
            # `join()` is unconditional and unbounded, so a worker that never
            # comes back would block every later flush() forever — including
            # direct callers of the public method, who have no deadline of their
            # own. A counter plus a condition lets flush() wake the moment the
            # queue drains, and give up on a deadline or a dead worker.
            self._pending = 0
            self._drained = threading.Condition()
            # daemon=True with NO atexit hook of our own, and no shutdown
            # sentinel: `_flush_all_providers` is the single bounded shutdown
            # drain path, and the worker just dies with the process.
            # Constructed lazily on the first dispatched event (`_get_providers`
            # is only reached from `_dispatch`), so a run that sends nothing —
            # `comfy --help`, shell completion, no consent — never starts it.
            self._worker = threading.Thread(target=self._run, daemon=True, name="mixpanel-telemetry")
            try:
                self._worker.start()
            except RuntimeError as e:
                # "can't start new thread" under thread/FD/memory pressure. A
                # telemetry-side resource problem must not become a user-visible
                # provider-construction failure (`_get_providers` reports those
                # with logging.warning, i.e. on the user's stderr), so degrade to
                # an inert provider instead of letting it escape.
                _log_telemetry_debug(f"could not start the mixpanel worker; disabling mixpanel telemetry: {e}")
                self.client = None
        self.enabled = self.client is not None

    def _mark_done(self) -> None:
        """Retire one dequeued event and wake `flush()` once nothing is left."""
        try:
            with self._drained:
                self._pending -= 1
                if self._pending <= 0:
                    self._drained.notify_all()
        except BaseException:  # noqa: BLE001  # pragma: no cover - defensive
            pass

    def _run(self) -> None:
        """Drain the queue forever, one send at a time.

        A single worker over a single FIFO preserves per-process event ordering.
        Accepted semantic shift: mixpanel-python stamps an event's `time` when
        `client.track()` runs, which moves from call time to dequeue time —
        sub-second skew in practice, no dashboard impact.
        """
        while True:
            # The WHOLE body is guarded, not just the send: nothing detects or
            # restarts this thread, so anything that escapes wedges telemetry for
            # the rest of the process — every later event silently dropped and
            # every flush() burning its full deadline. BaseException rather than
            # Exception because a MemoryError or SystemExit out of the SDK would
            # do exactly that.
            item = None
            event_name = "<unknown>"
            try:
                item = self._queue.get()
                event_name, distinct_id, properties = item
                self.client.track(distinct_id=distinct_id, event_name=event_name, properties=properties)
            except BaseException as e:  # noqa: BLE001
                # debug, not warning: a telemetry failure must never surface on
                # the user's stderr (cf. the urllib3/posthog silencing above).
                _log_telemetry_debug(f"Failed to send mixpanel event {event_name}: {e}")
            finally:
                # In `finally`, and keyed on having actually dequeued something,
                # so a raising send can't leave flush() waiting on an event that
                # is never retired — nor retire one that was never taken.
                if item is not None:
                    self._mark_done()

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None:
        if self.client is None or distinct_id is None:
            return
        # deepcopy, not dict(): `@track_command` fires its event *before* running
        # the wrapped body, and serialization now happens on the worker instead of
        # inline, so a shallow copy leaves nested values (Typer multi-value
        # options, feedback score dicts) aliased to objects the body can still
        # mutate — shipping post-mutation contents, or racing mixpanel's
        # json.dumps into "dictionary changed size during iteration" and dropping
        # the event. Copying here restores the old snapshot-at-call-time
        # semantics; a value deepcopy can't handle falls back to the shallow copy.
        try:
            payload = copy.deepcopy(properties)
        except Exception:  # noqa: BLE001
            payload = dict(properties)
        with self._drained:
            try:
                self._queue.put_nowait((event_name, distinct_id, payload))
            except queue.Full:
                _log_telemetry_debug(f"mixpanel queue full; dropping event {event_name}")
                return
            # Counted under the same lock the worker takes to retire an event, so
            # a send that completes before we return here can't decrement first.
            self._pending += 1

    def flush(self) -> None:
        if self.client is None:
            return
        # Waits for the queue to drain completely, including the send already in
        # flight — but bounded, and abandoned outright if the worker is gone.
        # `_flush_all_providers` supplies its own deadline at exit; this one is
        # for every other caller of what is, after all, a public method.
        deadline = time.monotonic() + _FLUSH_DEADLINE_SECONDS
        with self._drained:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._worker.is_alive():
                    _log_telemetry_debug(f"mixpanel flush gave up with {self._pending} event(s) still queued")
                    return
                # Short waits so a worker that dies mid-drain is noticed promptly
                # rather than at the deadline.
                self._drained.wait(timeout=min(remaining, 0.1))


class PostHogProvider:
    _STANDARD_PROPERTIES = {
        "environment": "cli",
        "surface": "cli",
        "source": "cli",
        "trigger_source": "cli",
    }

    def __init__(self, token: str, host: str):
        self.client: Posthog | None = None
        self.enabled = False
        if not token:
            return
        # Imported here, not at module scope: see `_get_providers`.
        from posthog import Posthog

        # disable_geoip=False lets PostHog enrich events with IP-derived location.
        # max_retries/timeout tighten the consumer drain budget from the posthog
        # 7.x defaults (3 × 15s ≈ 50s worst case) to ~21s, so the atexit flush
        # can't linger on a blackholed endpoint after the terminal envelope is
        # already on stdout.
        self.client = Posthog(project_api_key=token, host=host, disable_geoip=False, max_retries=1, timeout=10)
        # Posthog's constructor registers its own atexit.register(self.join),
        # which runs self.join() synchronously on the main thread at shutdown —
        # independently of _flush_all_providers and NOT bounded by its daemon
        # deadline. Against a blackholed endpoint that join can still block ~21s
        # after the terminal envelope is on stdout, defeating this change. Drop it
        # so our bounded flush is the only shutdown drain path.
        atexit.unregister(self.client.join)
        self.enabled = True

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None:
        if not self.enabled or self.client is None or distinct_id is None:
            return
        merged = {**self._STANDARD_PROPERTIES, **properties}
        # Membership check uses the canonical (unprefixed) name; the prefix is
        # cosmetic to the PostHog taxonomy and applied only at capture time.
        if event_name in EXECUTION_EVENTS and "tracing_id" in merged:
            merged.setdefault("workflow_run_id", merged["tracing_id"])
        self.client.capture(event=f"{POSTHOG_EVENT_PREFIX}{event_name}", distinct_id=distinct_id, properties=merged)

    def flush(self) -> None:
        if self.client is None:
            return
        # posthog-python ships asynchronously; without flush, short-lived CLI invocations silently drop in-flight events
        self.client.flush()


# Built on the first send, not at import. `mixpanel` pulls in pydantic (and thus
# the compiled pydantic_core extension) for its feature-flags module, and importing
# it here meant every `comfy` process held that .pyd open for its whole lifetime.
# On Windows a loaded DLL cannot be replaced, so `comfy install` — which shells out
# to `uv pip install` against its own interpreter's environment — could not upgrade
# pydantic_core and died with "Access is denied (os error 5)", leaving the package
# half-removed and every later `comfy` invocation unable to import it.
# Deferring the import means a run that never sends telemetry (no consent, or
# DO_NOT_TRACK) never loads the extension at all. See tests/comfy_cli/test_tracking_lazy_import.py.
PROVIDERS: list[TelemetryProvider] | None = None

# Building at import used to be single-shot courtesy of the import lock; keep that
# guarantee now that it happens on demand. Racing builds would strand a PostHog
# client — and its unflushed queue — in the list that lost.
_PROVIDERS_LOCK = threading.Lock()


def _get_providers() -> list[TelemetryProvider]:
    global PROVIDERS
    # Shell tab-completion (fish/bash/zsh) resolves the command tree without ever
    # running a command, so nothing is sent — don't stand up (or even import) the
    # telemetry SDKs on that inert path (GitHub #506). Today's lazy construction
    # already keeps completion inert because _dispatch never runs there; this makes
    # that guarantee explicit and independent of *when* providers get built.
    # Returned uncached so a real invocation is never affected.
    if _in_shell_completion():
        return []
    if PROVIDERS is None:
        with _PROVIDERS_LOCK:
            if PROVIDERS is None:
                # Construction runs the deferred SDK imports, so it can fail on exactly
                # the broken dependency tree this deferral exists to avoid (a half-removed
                # pydantic_core raises ImportError). Telemetry is best-effort: degrade to a
                # no-op rather than take the user's command down with us. Each provider is
                # built independently so one unusable SDK doesn't silence the other, and the
                # result is cached either way — including [] — so a doomed import isn't
                # retried on every later event.
                built = []
                for name, factory in (
                    ("MixpanelProvider", lambda: MixpanelProvider(MIXPANEL_TOKEN)),
                    ("PostHogProvider", lambda: PostHogProvider(_resolve_posthog_token(), POSTHOG_HOST)),
                ):
                    try:
                        built.append(factory())
                    except Exception as e:  # noqa: BLE001
                        logging.warning(f"Failed to initialize telemetry provider {name}, skipping it: {e}")
                PROVIDERS = built
    return PROVIDERS


app = typer.Typer()


@app.command(help="Opt in to anonymous usage analytics.")
def enable():
    """Opt in to anonymous usage analytics."""
    init_tracking(True)
    typer.echo("Tracking is now enabled.")


@app.command(help="Opt out of anonymous usage analytics.")
def disable():
    """Opt out of anonymous usage analytics."""
    init_tracking(False)
    typer.echo("Tracking is now disabled.")


def _dispatch(
    event_name: str,
    properties: dict[str, Any],
    *,
    distinct_id: str | None,
    mixpanel_name: str | None = None,
    caller_kind: str | None = None,
):
    """Fan an event out to every provider. Enriches with cli_version/tracing_id/caller_kind.

    This is the shared send path; callers above own the gating (consent for
    passive telemetry, env-only for feedback).

    ``caller_kind`` lands on EVERY event (execution_*, partner_nodes_detected,
    feedback, …) — that is the point: it makes agent-vs-human analytics possible
    across the whole stream. Purely additive, so no existing dashboard breaks.
    Defaults to the full label; a caller that ships on a weaker consent basis
    passes the narrowed one (see ``_intrinsic_caller_kind``).
    """
    properties = {
        **properties,
        "cli_version": cli_version,
        "tracing_id": tracing_id,
        "caller_kind": caller_kind if caller_kind is not None else _caller_kind,
    }
    for provider in _get_providers():
        provider_event_name = (
            mixpanel_name if (mixpanel_name is not None and isinstance(provider, MixpanelProvider)) else event_name
        )
        try:
            provider.track(provider_event_name, distinct_id=distinct_id, properties=dict(properties))
        except Exception as e:
            logging.warning(f"Failed to track event via {type(provider).__name__}: {e}")


def track_event(event_name: str, properties: Any = None, *, mixpanel_name: str | None = None):
    """Fire ``event_name`` to every enabled telemetry provider.

    ``mixpanel_name``, if supplied, overrides the event name on the Mixpanel pipe only — used to keep
    legacy Mixpanel event names while PostHog receives the canonical name.
    """
    if _telemetry_disabled_by_env():
        return
    if properties is None:
        properties = {}
    logging.debug(f"tracking event called with event_name: {event_name} and properties: {properties}")
    enable_tracking = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if not enable_tracking and not _session_only_tracking:
        return

    _dispatch(event_name, properties, distinct_id=user_id, mixpanel_name=mixpanel_name)


def _ensure_user_id(*, persist: bool = True) -> str:
    """Return a distinct_id. Persists a generated anonymous id only when
    ``persist`` is True (consent on). For an opted-out, user-initiated action
    we still attach an ephemeral id but never write durable identity to disk.
    """
    global user_id
    if user_id:
        return user_id
    existing = config_manager.get(constants.CONFIG_KEY_USER_ID)
    if existing:
        user_id = existing
        return user_id
    new_id = str(uuid.uuid4())
    if persist:
        user_id = new_id
        try:
            config_manager.set(constants.CONFIG_KEY_USER_ID, new_id)
        except OSError:
            pass
    return new_id


def submit_feedback(message: str = "", *, scores: dict[str, str | None] | None = None) -> bool:
    """Send user feedback to telemetry (PostHog + Mixpanel) as ``feedback_submitted``.

    Unlike passive command telemetry, feedback is an explicit, user-initiated
    action — so it is NOT gated on the consent flag. Only the hard env opt-out
    (``DO_NOT_TRACK`` / ``COMFY_NO_TELEMETRY``) suppresses it. Returns False
    without sending when opted out or when there's nothing to send, so the
    caller can tell the user rather than silently drop their words. Fail-fast:
    no on-disk queue, no retry — best-effort delivery.
    """
    if _telemetry_disabled_by_env():
        return False
    properties: dict[str, Any] = {}
    if message:
        properties["message"] = message
    if scores:
        properties.update({k: v for k, v in scores.items() if v is not None})
    if not properties:
        return False
    consented = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    # Narrowed caller_kind: this is the one path that sends without passive
    # consent, so it carries the closed set of intrinsic kinds rather than a
    # free-text COMFY_USER_AGENT label.
    _dispatch(
        "feedback_submitted",
        properties,
        distinct_id=_ensure_user_id(persist=bool(consented)),
        caller_kind=_intrinsic_caller_kind(),
    )
    return True


def submit_agent_review(summary: str = "", *, properties: dict[str, Any] | None = None) -> bool:
    """Send an agent-authored summary of how the session went as ``agent_review_submitted``.

    Distinct from :func:`submit_feedback`: this is the agent's assessment, not
    the user's words, so it is treated like passive telemetry — fully
    consent-gated. If the user opted out by ANY means (env opt-out, or no
    consent), nothing is sent and this returns False. No queue, no retry.
    """
    if not _consent_enabled():
        return False
    payload: dict[str, Any] = {}
    if summary:
        payload["summary"] = summary
    if properties:
        payload.update({k: v for k, v in properties.items() if v is not None})
    if not payload:
        return False
    _dispatch("agent_review_submitted", payload, distinct_id=user_id)
    return True


def filter_command_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop untrackable kwargs (see ``_is_trackable``), redact sensitive values
    (see ``_is_sensitive``), and strip credentials embedded in URL values
    (see ``_scrub_value``)."""
    return {
        k: ("<redacted>" if v is not None else None) if _is_sensitive(k) else _scrub_value(v)
        for k, v in kwargs.items()
        if _is_trackable(k, v)
    }


def track_command(sub_command: str | None = None):
    """
    A decorator factory that logs the command function name and selected arguments when it's called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            command_name = f"{sub_command}:{func.__name__}" if sub_command is not None else func.__name__
            filtered_kwargs = filter_command_kwargs(kwargs)
            logging.debug(f"Tracking command: {command_name} with arguments: {filtered_kwargs}")
            track_event(command_name, properties=filtered_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def prompt_tracking_consent(skip_prompt: bool = False, default_value: bool = False):
    global _session_only_tracking, user_id

    # Env-var opt-out short-circuits everything below: no prompt, no
    # auto-enable in non-TTY, no user_id persistence. Per-process only —
    # the on-disk consent flag is left untouched so a later run without
    # the env var still gets the normal prompt path.
    if _telemetry_disabled_by_env():
        return

    if _session_only_tracking:
        return

    tracking_enabled = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if tracking_enabled is not None:
        return

    if skip_prompt:
        init_tracking(default_value)
        return

    # Non-interactive sessions (pipes, CI, agents) default to no tracking
    # until the user explicitly consents via an interactive terminal.
    # Persist a stable anonymous user_id so a later interactive consent
    # prompt can reuse it, but do NOT auto-enable telemetry — that would
    # violate the DO_NOT_TRACK convention spirit for OSS tooling.
    # Guarded probes: this runs from the main Typer callback, so a detached /
    # `pythonw` stdio pair must degrade to "non-interactive" rather than raise
    # and take the command down. See `caller.stream_is_tty`.
    if not stream_is_tty(getattr(sys, "stdin", None)) or not stream_is_tty(getattr(sys, "stdout", None)):
        if user_id is None:
            user_id = str(uuid.uuid4())
            try:
                config_manager.set(constants.CONFIG_KEY_USER_ID, user_id)
            except OSError:
                pass
        return

    enable_tracking = ui.prompt_confirm_action("Do you agree to enable tracking to improve the application?", False)
    init_tracking(enable_tracking)


def init_tracking(enable_tracking: bool):
    """
    Initialize the tracking system by setting the user identifier and tracking enabled status.
    """
    global user_id
    logging.debug(f"Initializing tracking with enable_tracking: {enable_tracking}")
    config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, str(enable_tracking))
    if not enable_tracking:
        return

    curr_user_id = config_manager.get(constants.CONFIG_KEY_USER_ID)
    logging.debug(f'User identifier for tracking user_id found: {curr_user_id}."')
    if curr_user_id is None:
        curr_user_id = str(uuid.uuid4())
        config_manager.set(constants.CONFIG_KEY_USER_ID, curr_user_id)
        logging.debug(f'Setting user identifier for tracking user_id: {curr_user_id}."')
    user_id = curr_user_id

    # Note: only called once when the user interacts with the CLI for the
    #  first time iff the permission is granted.
    install_event_triggered = config_manager.get_bool(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED)
    if not install_event_triggered:
        logging.debug("Tracking install event.")
        config_manager.set(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED, "True")
        track_event("install")


def _flush_one(provider: TelemetryProvider) -> None:
    try:
        provider.flush()
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Failed to flush telemetry provider {type(provider).__name__}: {e}")


_FLUSH_DEADLINE_SECONDS = 5.0


def _flush_all_providers() -> None:
    # Deliberately reads PROVIDERS rather than calling _get_providers(): a run that
    # never sent anything has nothing to drain, and constructing providers from an
    # atexit hook would import the SDKs we just went to the trouble of deferring.
    # Taking the lock (without building) makes an in-flight build resolve first —
    # otherwise we could read None mid-construction and exit without draining the
    # racing thread's PostHog queue. Released before flushing so a slow network
    # drain doesn't block a concurrent send.
    with _PROVIDERS_LOCK:
        providers = PROVIDERS
    if providers is None:
        return
    # Telemetry is best-effort by contract: a blackholed endpoint (accepts TCP,
    # never responds) must never let this atexit hook wedge every consumer of the
    # CLI's stdout after the terminal envelope is already emitted.
    # Start every provider's flush in a daemon thread, then join them all against
    # a SINGLE shared deadline so total exit delay stays ~5s regardless of how
    # many providers there are (a per-provider join would make it 5s × N).
    # Dropping in-flight events beats hanging the process. t.start()/t.join() are
    # wrapped defensively so nothing this hook does can raise and print a
    # traceback to stderr after the terminal envelope.
    deadline = time.monotonic() + _FLUSH_DEADLINE_SECONDS
    threads: list[tuple[threading.Thread, TelemetryProvider]] = []
    for provider in providers:
        t = threading.Thread(target=_flush_one, args=(provider,), daemon=True)
        try:
            t.start()
        except RuntimeError as e:  # e.g. a thread-creation race during shutdown
            logging.warning(f"could not start telemetry flush for {type(provider).__name__}: {e}")
            continue
        threads.append((t, provider))
    for t, provider in threads:
        try:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        except RuntimeError as e:  # pragma: no cover - defensive
            logging.warning(f"telemetry flush join failed for {type(provider).__name__}: {e}")
            continue
        if t.is_alive():
            # debug, not warning: this fires purely because a telemetry endpoint
            # is slow or blackholed, and it fires at exit — i.e. it would print
            # to the user's stderr *after* the terminal envelope. That is the one
            # thing this module refuses to do for a telemetry failure. It was
            # unreachable for Mixpanel while flush() was a no-op; it isn't now
            # that flush() actually drains a queue.
            logging.debug(f"telemetry flush timed out for {type(provider).__name__}; dropping in-flight events")


def flush_for_hard_exit() -> None:
    """Drain telemetry before an `os._exit`, which skips atexit handlers.

    `comfy launch` terminates through `os._exit` on both its background-success
    and failure paths, so `_flush_all_providers` never runs there. That was
    harmless while MixpanelProvider sent inline from `track()` — the `launch`
    event was already delivered before the command body ran. Now that dispatch is
    queue-and-drain the event is still sitting in the queue at that
    point, so those paths have to drain explicitly or drop it every time.

    Bounded by the same `_FLUSH_DEADLINE_SECONDS` budget as the atexit hook, and
    best-effort: nothing it does may keep the caller from exiting.
    """
    try:
        _flush_all_providers()
    except BaseException:  # noqa: BLE001  # pragma: no cover - defensive
        pass


atexit.register(_flush_all_providers)
