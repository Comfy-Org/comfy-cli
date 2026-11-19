# ADR-BUILD-VALIDATION-0015: Let `validate --remote` Ask Model Hosts for a Hash

Date: 2026-11-11

## Status

Proposed

Relation: Builds on
[ADR-BUILD-VALIDATION-0008](BUILD-VALIDATION-0008-let-a-dry-run-check-public-pins-with-the-builder.md).
That record widens what `push --dry-run` contacts; this one widens what
`validate --remote` contacts, and 0008 stays as it is. No error code is added.

## Context

`comfy build validate --remote` used to contact only the builder: it read the
model folder list and looked each model's filename up in the public model
catalog. A model that names its own `sourceUri` and `sha256` got the same
filename lookup, which often reads `none_found` for a link that is fine.

The link's host can say more. Hugging Face states the sha256 of a stored file
as `X-Linked-Etag` on the redirect to its CDN, and the builder's own verifier
reads that header. Asking the host lets `validate` say whether the declared
hash matches before a cut downloads the file.

That means `validate --remote` now sends requests to every host a spec names,
not only the builder. Scripts and agents rely on what a command contacts, so
that is recorded here.

## Decision

`comfy build validate --remote` may send one `HEAD` to a model's `sourceUri`:

- **When.** Only with `--remote`, and only for a model that has both a
  `sourceUri` and a `sha256`. Without `--remote`, `validate` sends nothing.
- **Https only.** A link that is not `https://` with a host is never opened. A
  `source: local` entry keeps its link on the wire, so this check is what keeps
  a credential in an `http://` link off the network.
- **No userinfo.** A link with a user or password before its host
  (`https://TOKEN@host/...`) is never opened either. Through an HTTP proxy,
  urllib puts that userinfo in its `CONNECT` line, which goes to the proxy in
  the clear before TLS starts. urllib percent-decodes the host first, so
  `TOKEN%40host` would go out as `TOKEN@host`: a host with any `@` or `%` in
  it is refused. The model gets the filename lookup, and no message names the
  link.
- **What is sent.** The `HEAD` for the URL as written, with urllib's default
  headers. No builder token, Cloud key, cookie or other credential is added;
  the only credential that travels is one the URL itself carries (a Civitai
  `?token=`), and only to that URL's host.
- **No redirects.** The request goes through `comfy_cli.http.no_redirect_urlopen`.
  A redirect is read for its headers and not followed, so a credential in the
  URL never reaches a second host.
- **What is read.** Only `X-Linked-Etag`, and only when it is 64 hex
  characters. A plain `ETag` is never read as a hash, as in the builder's
  verifier. The entry reads `declared_ok` or `declared_mismatch`, with the
  host's hash in `sourceSha256`.
- **Time budget.** Each `HEAD` may take 5 seconds. All of them together may take
  15 seconds; each gets what is left of that, up to its own 5. The limit counts
  the whole request, from connect to the last header, not each socket read:
  urllib's timeout applies per read, so a host that sends its headers a little
  at a time would never trip it. Each `HEAD` runs in a daemon thread that
  `validate` waits on for its share of the budget; one still running then is
  abandoned as timed out, and cannot keep the command from exiting. Once the
  budget is spent, no more links are asked, and the command says how many
  models got the filename lookup without a link check.
- **Fallback.** A link that is not asked, states no hash, refuses, times out or
  fails in any other way gets the filename lookup, as before this record. A
  link check never fails the command.

The `--remote` help says the `HEAD`s go to the model hosts and share the
15-second budget.

## Alternatives considered

- **Ask the builder to check the links.** It would keep `validate` talking to
  one host, but the builder has no endpoint for it, and a link with a token
  would then leave the machine to the builder as well.
- **Follow the redirect and hash the file.** That downloads the whole model,
  often gigabytes, to answer a validation.
- **No time budget, a timeout per link only.** A spec of 40 models on hosts that
  do not answer waited 400 seconds, which an agent's command timeout reads as a
  hang.
- **Probe in parallel.** It would cut the wait, but still contacts every host,
  and the budget alone keeps the command short.
- **Check the budget only between `HEAD`s.** A host that trickles its headers
  keeps one `HEAD` open well past the budget, since urllib's timeout is per read.
- **Strip the userinfo and ask anyway.** The host would then answer for a
  different request than the one the builder makes, and the user or password is
  dropped silently; the filename lookup is the honest fallback.

## What we reject

- Adding any credential the URL does not carry.
- Following a redirect from a model host.
- Opening a link that is not https, or one with userinfo before its host,
  written out or percent-encoded.
- Reading a plain `ETag` as a sha256.
- Letting a link check that fails or runs out of time fail `validate`.

## Consequences

### Positive

- A declared link whose hash matches reads `declared_ok`, and a wrong hash is
  caught before a cut downloads the file.
- `validate --remote` takes at most about 15 seconds on links, however many
  models the spec has.

### Negative

- `validate --remote` now contacts hosts other than the builder: Hugging Face,
  Civitai, or whatever a spec names. A network that allows only the builder
  sees those requests time out and fall back.
- A spec with many slow links gets the link check on the first few only.

## Validation

- `tests/comfy_cli/command/test_build_validate.py`: a matching and a differing
  `X-Linked-Etag` read `declared_ok` and `declared_mismatch`; a link that does
  not answer gets the filename lookup; a plain `ETag` gives no verdict; a link
  that is not https, or that has userinfo before its host, written out or
  percent-encoded, is never opened;
  40 links on hosts that never answer stop at the budget and every model gets
  the filename lookup; hosts that trickle their headers past each read's
  timeout are cut off at the budget.

## Revisit triggers

- The builder gains an endpoint that checks a link's hash. `validate` could then
  contact only the builder again.
- Another major host states a sha256 in a header other than `X-Linked-Etag`.
- Users report the 15-second budget cutting off checks they need.
