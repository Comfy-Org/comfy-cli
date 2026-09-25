from __future__ import annotations

import ipaddress
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import unquote

from typing_extensions import assert_never

from comfy_cli.command.build_paths import BuildPaths, resolve_local_path
from comfy_cli.command.build_spec import SPEC_SCHEMA, BuildSpecInvalidError, JsonObject, JsonValue

_AUTHORING_FIELDS: Final = frozenset({"source", "localPath", "localDigest", "localSizeBytes"})
MODEL_SOURCES: Final = ("blobId", "sourceUri")
NODE_SOURCES: Final = ("blobId", "registryVersion", "repository")
MODEL_RESOLVE_BATCH_SIZE: Final = 32

# comfy-builder's own model rules (``definition.validateModels`` and
# ``common.ValidModelDir``), so a spec its release cut would refuse is refused here,
# before any upload, with every problem at once. The reasons are the builder's words,
# so what this prints reads the same as what a save or a cut would say.
_SAFE_SEGMENT: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MAX_MODEL_DIR: Final = 255
_RESERVED_MODEL_ROOTS: Final = frozenset({"configs", "custom_nodes"})
_WINDOWS_RESERVED: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)
_MODEL_DIR_REASON: Final = "must be a model directory under models/ (e.g. checkpoints, insightface/models/antelopev2)"

# Go's ``url.Parse``, as far as the builder's two link rules read it (``_go_url``).
# ``_GO_SPACE`` is what ``strings.TrimSpace`` trims: ``str.strip`` also trims
# \x1c-\x1f, which Go keeps and then refuses as control characters.
_GO_SPACE: Final = (
    "\t\n\v\f\r \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200B))) + "\u2028\u2029\u202f\u205f\u3000"
)
_URL_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f]")
# A link's userinfo, which a label leaves out with its query and fragment.
_LINK_USERINFO: Final = re.compile(r"^((?:[A-Za-z][A-Za-z0-9+.-]*:)?//)[^/?#]*@")
_URL_BAD_ESCAPE: Final = re.compile(r"%(?![0-9A-Fa-f]{2})")
_URL_SCHEME: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*(?=:)")
_URL_PORT: Final = re.compile(r"(:[0-9]*)?")
_URL_USERINFO: Final = re.compile(r"[A-Za-z0-9\-._:~!$&'()*+,;=%@]*")
# The ASCII a host may hold unescaped; ':' and '[' are further bound by the port
# and IPv6 rules.
_HOST_SAFE: Final = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:[]<>\"")
_HEX: Final = frozenset("0123456789abcdefABCDEF")


def _go_trim(value: str) -> str:
    """``strings.TrimSpace``, which the builder's rules trim with."""
    return value.strip(_GO_SPACE)


class ModelLookupState(str, Enum):
    CANDIDATE_FOUND = "candidate_found"
    NONE_FOUND = "none_found"
    LOOKUP_ERROR = "lookup_error"
    NOT_LOOKUPABLE = "not_lookupable"


@dataclass(frozen=True, slots=True)
class ModelLookup:
    index: int
    filename: str | None
    state: ModelLookupState
    candidates: tuple[JsonObject, ...] = ()
    error: str | None = None

    def as_json(self) -> JsonObject:
        result: JsonObject = {
            "entry": f"definition.models[{self.index}]",
            "filename": self.filename,
            "state": self.state.value,
        }
        if self.candidates:
            result["candidates"] = [deepcopy(candidate) for candidate in self.candidates]
        if self.error is not None:
            result["error"] = self.error
        return result


class ModelResolver(Protocol):
    def __call__(self, filenames: list[str]) -> list[JsonObject]: ...


@dataclass(frozen=True, slots=True)
class LocalPathContext:
    location: str
    root: Path
    kind: Literal["model", "node"]


def _entries(definition: JsonObject, collection: str) -> list[JsonObject]:
    value = definition.get(collection, [])
    if not isinstance(value, list):
        raise BuildSpecInvalidError(f"definition.{collection} must be a list")
    entries: list[JsonObject] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise BuildSpecInvalidError(f"definition.{collection}[{index}] must be a mapping")
        entries.append(entry)
    return entries


def _set_sources(entry: JsonObject, fields: tuple[str, ...], *, location: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for field in fields:
        value = entry.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise BuildSpecInvalidError(f"{location}.{field} must be a string or null")
        if _go_trim(value):
            sources[field] = value
    return sources


def _project_model(entry: JsonObject, *, location: str) -> JsonObject:
    sources = _set_sources(entry, MODEL_SOURCES, location=location)
    projected = deepcopy(entry)
    for field in (*_AUTHORING_FIELDS, *MODEL_SOURCES):
        projected.pop(field, None)
    for winner in MODEL_SOURCES:
        if winner in sources:
            projected[winner] = sources[winner]
            break
    return projected


def _project_node(entry: JsonObject, *, location: str) -> JsonObject:
    sources = _set_sources(entry, NODE_SOURCES, location=location)
    projected = deepcopy(entry)
    for field in (*_AUTHORING_FIELDS, *NODE_SOURCES):
        projected.pop(field, None)
    for winner in NODE_SOURCES:
        if winner not in sources:
            continue
        projected[winner] = sources[winner]
        if winner in {"blobId", "registryVersion"}:
            projected.pop("gitRef", None)
            projected.pop("commit", None)
        break
    return projected


def project_wire_definition(definition: JsonObject) -> JsonObject:
    """Return the exclusion-based builder wire copy using D-I source precedence."""
    projected = deepcopy(definition)
    if "models" in definition:
        projected["models"] = [
            _project_model(entry, location=f"definition.models[{index}]")
            for index, entry in enumerate(_entries(definition, "models"))
        ]
    if "customNodes" in definition:
        projected["customNodes"] = [
            _project_node(entry, location=f"definition.customNodes[{index}]")
            for index, entry in enumerate(_entries(definition, "customNodes"))
        ]
    return projected


def _required_string(entry: JsonObject, field: str, *, location: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BuildSpecInvalidError(f"{location}.{field} must be a non-empty string")
    return value


def _optional_string(entry: JsonObject, field: str, *, location: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BuildSpecInvalidError(f"{location}.{field} must be a string or null")
    return value


def _validate_local_path(entry: JsonObject, context: LocalPathContext) -> None:
    local_path = _optional_string(entry, "localPath", location=context.location)
    source = _optional_string(entry, "source", location=context.location)
    if local_path is None:
        if source == "local":
            raise BuildSpecInvalidError(f"{context.location}.localPath is required when source is 'local'")
        return
    if not local_path.strip():
        raise BuildSpecInvalidError(f"{context.location}.localPath must be a non-empty string")
    resolved = resolve_local_path(context.root, local_path, entry=context.location)
    match context.kind:
        case "model":
            correct_kind = resolved.is_file()
            expected = "file"
        case "node":
            correct_kind = resolved.is_dir()
            expected = "directory"
        case unreachable:
            assert_never(unreachable)
    if not correct_kind:
        raise BuildSpecInvalidError(f"{context.location}.localPath must resolve to a {expected}: {local_path!r}")


def _validate_authoring_definition(definition: JsonObject, paths: BuildPaths) -> None:
    schema = definition.get("schema")
    if schema is not None and not isinstance(schema, str):
        raise BuildSpecInvalidError("definition.schema must be a string or null")
    for index, model in enumerate(_entries(definition, "models")):
        location = f"definition.models[{index}]"
        _required_string(model, "type", location=location)
        _optional_string(model, "filename", location=location)
        _validate_local_path(model, LocalPathContext(location, paths.models_dir, "model"))
    for index, node in enumerate(_entries(definition, "customNodes")):
        location = f"definition.customNodes[{index}]"
        _required_string(node, "name", location=location)
        _validate_local_path(node, LocalPathContext(location, paths.custom_nodes_dir, "node"))


def _validate_wire_sources(original: JsonObject, projected: JsonObject, collection: str) -> None:
    fields = MODEL_SOURCES if collection == "models" else NODE_SOURCES
    originals = _entries(original, collection)
    wire_entries = _entries(projected, collection)
    for index, (authoring_entry, wire_entry) in enumerate(zip(originals, wire_entries)):
        effective = _set_sources(wire_entry, fields, location=f"definition.{collection}[{index}]")
        if len(effective) > 1:
            raise BuildSpecInvalidError(f"definition.{collection}[{index}] has multiple effective builder sources")
        if effective or authoring_entry.get("source") == "local":
            continue
        raise BuildSpecInvalidError(f"definition.{collection}[{index}] has no effective builder source")


def _valid_model_dir(value: str, directories: frozenset[str] | None) -> bool:
    """``common.ValidModelDir``: a vetted directory, or a relative path that can only
    land inside models/. A case variant of a vetted directory ("Loras") is a typo, and
    only the builder's list can tell one, so without *directories* it passes here and
    the save warns about it instead."""
    if directories and value in directories:
        return True
    if not value or len(value) > _MAX_MODEL_DIR:
        return False
    if directories and value.lower() in {directory.lower() for directory in directories}:
        return False
    segments = value.split("/")
    if segments[0].lower() in _RESERVED_MODEL_ROOTS:
        return False
    for segment in segments:
        if not _SAFE_SEGMENT.fullmatch(segment) or segment.endswith("."):
            return False
        if segment.partition(".")[0].upper() in _WINDOWS_RESERVED:
            return False
    return True


def _valid_filename(value: str) -> bool:
    return _SAFE_SEGMENT.fullmatch(value) is not None and ".." not in value


def _go_url(value: str) -> tuple[str, str, str] | None:
    """Go's ``url.Parse(strings.TrimSpace(value))`` as far as the builder reads it:
    the lower-cased scheme, the host and the percent-decoded path, or None where Go
    returns an error. Written out because ``urlsplit`` accepts what Go refuses (a
    control character, a bad ``%`` escape, a port that is not a number) and leaves
    the path encoded, so ``m%2Esafetensors`` would have no extension."""
    rest, _, fragment = _go_trim(value).partition("#")
    if _URL_CONTROL.search(rest) or _URL_BAD_ESCAPE.search(fragment) or rest.startswith(":"):
        return None
    scheme = ""
    if match := _URL_SCHEME.match(rest):
        scheme, rest = match[0].lower(), rest[match.end() + 1 :]
    rest = rest.partition("?")[0]
    if not rest.startswith("/"):
        if scheme:
            return scheme, "", ""  # opaque, as "https:h.example/x" is: no host, no path
        if ":" in rest.partition("/")[0]:
            return None
    host = ""
    if rest.startswith("//") and (scheme or not rest.startswith("///")):
        authority, slash, path = rest[2:].partition("/")
        rest = slash + path
        parsed_host = _go_host(scheme, authority)
        if parsed_host is None:
            return None
        host = parsed_host
    if _URL_BAD_ESCAPE.search(rest):
        return None
    return scheme, host, unquote(rest, errors="surrogateescape")


def _go_host(scheme: str, authority: str) -> str | None:
    """Go's ``parseAuthority``: the host of *authority*, or None where Go refuses it."""
    userinfo, at, host = authority.rpartition("@")
    if at and (not _URL_USERINFO.fullmatch(userinfo) or _URL_BAD_ESCAPE.search(userinfo)):
        return None
    if "[" in host[1:]:
        return None
    if host.startswith("["):
        close = host.rfind("]")
        if close < 0 or not _URL_PORT.fullmatch(host[close + 1 :]):
            return None
        address, zoned, zone = host[1:close].partition("%25")
        if not _go_host_unescapes(address, zone=False) or not _go_host_unescapes(zoned + zone, zone=True):
            return None
        # netip.ParseAddr, which takes only an IPv6 address here: a zone after the
        # first "%" must not be empty, and may itself hold one.
        ip, percent, zone_name = unquote(address + zoned + zone).partition("%")
        if percent and not zone_name:
            return None
        try:
            ipaddress.IPv6Address(ip)
        except ValueError:
            return None
        return host
    # Go ends the host at its first ':' in an http(s) link, at the last in any other.
    colon = host.find(":") if scheme in ("http", "https") else host.rfind(":")
    if colon >= 0 and not _URL_PORT.fullmatch(host[colon:]):
        return None
    if not _go_host_unescapes(host, zone=False):
        return None
    return unquote(host, errors="surrogateescape")


def _go_host_unescapes(part: str, *, zone: bool) -> bool:
    """Whether Go's ``unescape`` takes *part* in host mode, or zone mode for an IPv6
    zone. An escape is two hex digits and, in a host, ``%25`` or a non-ASCII byte;
    in a zone, ``%25``, a space or a byte a host may hold unescaped."""
    index = 0
    while index < len(part):
        char = part[index]
        if char != "%":
            if char < "\x80" and char not in _HOST_SAFE:
                return False
            index += 1
            continue
        code = part[index + 1 : index + 3]
        if len(code) < 2 or not set(code) <= _HEX:
            return False
        byte = int(code, 16)
        refused = (byte != 0x20 and chr(byte) not in _HOST_SAFE) if zone else byte < 0x80
        if refused and code != "25":
            return False
        index += 3
    return True


def _https_url(value: str) -> bool:
    """``validHTTPSURL``: Go parses it, with an https scheme and a host."""
    parsed = _go_url(value)
    return parsed is not None and parsed[0] == "https" and bool(parsed[1])


def _lacks_extension(uri: str) -> bool:
    """Whether the name the builder derives from *uri* (Go's ``path.Base`` of its
    decoded path) has no extension, as a Civitai ``/api/download/models/<id>`` link
    does. A link Go cannot parse is False, as there."""
    parsed = _go_url(uri)
    if parsed is None or not parsed[2]:
        return False  # path.Base("") is ".", which has one
    base = parsed[2].rstrip("/").rsplit("/", 1)[-1] or "/"
    return "." not in base


def model_label(entry: JsonObject) -> str:
    """What names *entry* in a refusal: its filename, else its link, else its local
    path. A link is named without its query, fragment and userinfo, where a signed
    link or a Civitai ``?token=`` carries its credential."""
    for key in ("filename", "sourceUri", "localPath"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            if key == "sourceUri":
                return _LINK_USERINFO.sub(r"\1", value.strip().partition("#")[0].partition("?")[0])
            return value
    return ""


def _link_problem(entry: JsonObject, wire: JsonObject) -> tuple[str, str] | None:
    """The builder's two link rules on the ``sourceUri`` *wire* sends, as ``(field,
    reason)``."""
    uri = wire.get("sourceUri")
    if not isinstance(uri, str) or not _go_trim(uri):
        return None
    if not _https_url(uri):
        return "sourceUri", "must be an https URL"
    filename = entry.get("filename")
    if not (isinstance(filename, str) and _go_trim(filename)) and _lacks_extension(uri):
        return "filename", "sourceUri has no file extension; set an explicit filename"
    return None


def model_rule_problems(
    definition: JsonObject,
    projected: JsonObject,
    directories: frozenset[str] | None = None,
    *,
    kept_links: bool = False,
) -> list[dict[str, str]]:
    """Every model entry the builder's cut would refuse, as ``{field, reason,
    model}``. ``models[<n>]`` counts the spec as it is read, which sorts the models,
    so ``model`` names the entry (its filename, link or local path) for a person
    looking for it in a file they ordered themselves. A ``source: local`` entry's
    link and sha256 are left alone here: push replaces both with what it uploads.
    With *kept_links* only the link of each ``source: local`` entry is checked, for
    ``validate_kept_links``."""
    problems: list[dict[str, str]] = []
    model = ""

    def refuse(field: str, reason: str) -> None:
        problems.append({"field": field, "reason": reason, "model": model})

    for index, (entry, wire) in enumerate(zip(_entries(definition, "models"), _entries(projected, "models"))):
        location = f"definition.models[{index}]"
        model = model_label(entry)
        local = entry.get("source") == "local"
        if kept_links:
            if local and (link := _link_problem(entry, wire)):
                refuse(f"{location}.{link[0]}", link[1])
            continue
        model_type = entry.get("type")
        if isinstance(model_type, str) and not _valid_model_dir(model_type, directories):
            refuse(f"{location}.type", _MODEL_DIR_REASON)
        filename = entry.get("filename")
        if isinstance(filename, str) and _go_trim(filename) and not _valid_filename(filename):
            refuse(f"{location}.filename", "must be a safe filename")
        if local:
            continue
        if link := _link_problem(entry, wire):
            refuse(f"{location}.{link[0]}", link[1])
        sha256 = entry.get("sha256")
        if isinstance(sha256, str) and _go_trim(sha256) and not _SHA256.fullmatch(_go_trim(sha256).lower()):
            refuse(f"{location}.sha256", "must be a 64-character sha256")
    return problems


def validate_kept_links(definition: JsonObject) -> None:
    """The link rules on each ``source: local`` model of a definition ``prepare_push``
    reconciled. One whose file still matches its sha256 keeps its ``sourceUri`` and
    is not uploaded, so that link is what the builder reads."""
    _refuse(model_rule_problems(definition, project_wire_definition(definition), kept_links=True))


def _refuse(problems: list[dict[str, str]]) -> None:
    if problems:
        raise BuildSpecInvalidError(_problems_message(problems), issues=problems)


def _problems_message(problems: list[dict[str, str]]) -> str:
    count = len(problems)
    lines = [
        f"  {problem['field']}" + (f" ({problem['model']})" if problem["model"] else "") + f": {problem['reason']}"
        for problem in problems
    ]
    heading = f"{count} problem{'s' if count != 1 else ''} the builder would refuse this build for:"
    return "\n".join([heading, *lines])


def validate_local_build_spec(
    spec: JsonObject, paths: BuildPaths, *, model_directories: frozenset[str] | None = None
) -> JsonObject:
    """Run authoring validation, then return the validated normalized wire copy.

    *model_directories* is the builder's vetted list, when the caller has signed in
    and read it; it is what tells a case variant of a folder from a new one."""
    if spec.get("schema") != SPEC_SCHEMA:
        raise BuildSpecInvalidError(f"unsupported build spec schema {spec.get('schema')!r}; expected {SPEC_SCHEMA!r}")
    definition: JsonValue = spec.get("definition")
    if not isinstance(definition, dict):
        raise BuildSpecInvalidError("definition must be a mapping")
    _validate_authoring_definition(definition, paths)
    projected = project_wire_definition(definition)
    _validate_wire_sources(definition, projected, "models")
    _validate_wire_sources(definition, projected, "customNodes")
    _refuse(model_rule_problems(definition, projected, model_directories))
    return projected


def _lookup_result(index: int, filename: str, response: JsonValue | None) -> ModelLookup:
    location = f"definition.models[{index}]"
    if not isinstance(response, dict):
        return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned no result")
    echoed = response.get("filename")
    if echoed != filename:
        return ModelLookup(
            index,
            filename,
            ModelLookupState.LOOKUP_ERROR,
            error=f"lookup returned {echoed!r} for {location}",
        )
    error = response.get("error")
    if error is not None:
        if not isinstance(error, str):
            return ModelLookup(
                index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned a malformed error"
            )
        if error.strip():
            return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error=error)
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned malformed candidates")
    parsed: list[JsonObject] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return ModelLookup(
                index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned a malformed candidate"
            )
        parsed.append(candidate)
    state = ModelLookupState.CANDIDATE_FOUND if parsed else ModelLookupState.NONE_FOUND
    return ModelLookup(index, filename, state, candidates=tuple(parsed))


def lookup_public_model_sources(definition: JsonObject, resolver: ModelResolver) -> list[ModelLookup]:
    """Resolve lookupable model filenames in 32-item batches and restore spec order."""
    models = _entries(definition, "models")
    lookupable: list[tuple[int, str]] = []
    results: dict[int, ModelLookup] = {}
    for index, model in enumerate(models):
        filename = model.get("filename")
        if isinstance(filename, str) and filename.strip():
            lookupable.append((index, filename))
        else:
            results[index] = ModelLookup(index, None, ModelLookupState.NOT_LOOKUPABLE)

    for start in range(0, len(lookupable), MODEL_RESOLVE_BATCH_SIZE):
        batch = lookupable[start : start + MODEL_RESOLVE_BATCH_SIZE]
        response = resolver([filename for _, filename in batch])
        for offset, (index, filename) in enumerate(batch):
            item = response[offset] if offset < len(response) else None
            results[index] = _lookup_result(index, filename, item)
    return [results[index] for index in range(len(models))]
