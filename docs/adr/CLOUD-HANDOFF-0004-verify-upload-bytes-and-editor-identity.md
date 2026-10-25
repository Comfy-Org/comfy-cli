# ADR-CLOUD-HANDOFF-0004: Verify Upload Bytes and Editor Identity

Date: 2026-10-25

Relation: Builds on [ADR-AUTOMATION-0003](AUTOMATION-0003-one-versioned-machine-output-dialect.md),
[ADR-OPERATIONS-0002](OPERATIONS-0002-version-structured-workflow-operations.md), and
[ADR-DEPENDENCIES-0001](DEPENDENCIES-0001-delegate-unified-resolution-to-comfyui-manager.md).
The versioned output/error dialect, frozen operation vocabulary, and Manager's
ownership of unified dependency resolution remain unchanged.

## Status

Proposed

## Context

A local Run on Cloud handoff crosses several boundaries: file upload approval,
workspace ownership, model identity, deployment readiness, and the saved document
opened by a browser. A matching filename, successful upload, ready deployment, or
saved record alone does not establish a runnable workflow or a usable private
editor link. The frontend also accepts canvas formats with JavaScript-specific
refinements; a second handwritten Python subset can accept unusable handoffs or
reject valid saved documents.

The user approved exact-file model import and private document/environment link
support. They explicitly retained the general agent ledger and immediate recheck
contract for existing build/deployment uploads; a universal atomic upload-plan
protocol is outside this change.

## Decision

### Approval and model import

- The skill records each approved path, byte size, digest, workspace and
  destination. It reconciles the final API asset plan with the actual run cwd and
  roots immediately before upload. New files, changed bytes or another target
  require a new decision. This reduces races; it is not an atomic filesystem lock.
- The new `model upload` command requires an explicit workspace, named-file
  approval and expected SHA256. It creates a private temporary snapshot, verifies
  that digest, and uploads only that immutable snapshot. Capability/type/name
  checks occur before sending bytes. A dry run creates no model asset or job.
- Content identity and ownership are separate. Only an identical asset already
  owned by the authenticated workspace qualifies for reuse without byte upload.
  An identical public asset alone does not establish ownership. Different bytes
  at the destination name are a collision, not permission to overwrite.
- A successful import verifies the returned model folder and content hash and
  reports its actual loader path. Upload failure may have remote effects, so an
  uncertain result must not imply that an automatic retry is effect-free. A real
  job is still required before claiming the imported model runs.
- General build repair must establish a supported installation mechanism before
  cutting for a missing import. A constraint pins a package installed by another
  mechanism; it does not by itself install one. This adds handoff policy without
  reimplementing Manager's resolver.

### Read-only editor discovery

- `deploy ui` and `workflow ui` discover links without selecting an environment,
  opening a browser, uploading a file, modifying a workflow or starting compute.
  They use the existing versioned envelope/error registry, not a new dialect.
- Deployment links require the authenticated current workspace plus an exact
  ready, nondeleted deployment whose served release agrees with the backend's
  editor listing. The backend remains authoritative for membership, SSO, OAuth
  scope, rollout and selection eligibility. API-key job access does not prove
  browser-editor access. Credential precedence remains the shared CLI policy,
  including explicitly forwarded environment credentials; discovery does not
  weaken server authorization or embed credentials in links.
- Private workflow links require the literal server capability
  `private_workflow_links_enabled: true`. Absence or false is unsupported until
  the matching frontend release is deployed and the operator enables it.
- A private link targets the ordinary saved workflow UUID only after verifying
  the exact record/version/content and its matching normal
  `workflows/<record-name>.json` sidebar file. Duplicate, malformed, incompatible
  or mismatched evidence fails closed. A custom link also verifies the exact
  deployment and served release; a standard link explicitly selects the standard
  environment in the browser. The browser rechecks access and performs selection.
- An unavailable complete handoff returns a precise limitation and verified
  sidebar/environment/API fallback. A UUID fragment or guessed URL is not proof
  of document access. A public share requires its own publishing approval.

### Frontend consumer parity and URL dependency

- Bundle input JSON Schemas generated from the matching frontend consumer, with
  commit/file provenance. Restore only the input refinements that schema export
  cannot represent. Opaque metadata/widget objects are not traversed as nodes.
- Validate acceptance without normalizing or rewriting the user's graph. Compare
  both canvas versions against the actual frontend oracle corpus whenever its
  schemas/refinements change; retain valid unusual inputs and reject invalid
  graph fields at the same structural positions as the consumer.
- Use `ada-url>=4,<5` for WHATWG URL acceptance, including browser-equivalent
  Unicode scalar conversion, no implicit base and no network access. Parser
  unavailability fails closed. The CLI needs no Node runtime; Node is used only
  to regenerate source schemas and replay the frontend oracle during development.

## Alternatives considered

- Make every existing upload atomic through a universal manifest token. Deferred
  by explicit user choice; it requires a separate protocol spanning current
  build/deployment upload mechanisms. The new immutable model-upload snapshot is
  a bounded addition, not a claim that this protocol already exists elsewhere.
- Treat public visibility as workspace ownership, or filenames/sizes as model
  identity. Rejected because visibility and approximate metadata do not establish
  ownership or the approved bytes.
- Construct editor URLs from local ready status or metadata alone. Rejected
  because authorization, rollout, selected release and saved-file identity can
  disagree independently.
- Maintain a minimal handwritten Python canvas schema or approximate URL regex.
  Rejected because frontend acceptance includes structured refinements and
  WHATWG behavior. A Node subprocess at runtime was also rejected because it adds
  a second deployment/runtime requirement to the Python CLI.

## What we reject

Implicit upload/publishing approval, changed targets hidden behind a reused name,
silent lossy model substitution, guessed private links, weakening unrelated CQL
validation to admit imported models, and synthetic success presented as live
execution evidence. New output/error variants remain additive under
ADR-AUTOMATION-0003, and workflow edits do not bypass ADR-OPERATIONS-0002.

## Consequences

### Positive

Approved model bytes cannot change during the upload stream. Link discovery
exposes verified point-in-time identities and supports older releases with honest
fallbacks. Frontend changes have an executable acceptance-drift signal.

### Negative

Model upload needs temporary disk space for a full file snapshot. Read-only
identity checks add authenticated reads and can refuse when evidence is
incomplete. Ada adds a bounded compiled dependency and platform-wheel obligation.
Schema/corpus regeneration needs the matching frontend development environment.

### Known limits

General ledger/rehash checks retain filesystem race limits. Discovery cannot
promise that a deployment or permission remains unchanged until the link opens;
the browser/backend must recheck. Publication of the CLI and deployment/enablement
of matching frontend/backend capability are separate release actions. Local and
simulated tests do not establish original-workflow inference quality.

## Validation

- `tests/comfy_cli/command/test_model_upload.py` covers approval, workspace/type
  gating, immutable byte snapshots, collision/ownership behavior and uncertain
  HTTP results. `test_cloud_model_validation.py` retains unrelated validation
  failures and bounds authenticated exact-asset lookup.
- `test_deploy_ui.py` and `test_workflow_ui.py` cover authorization/target/release
  mismatch, disabled capability, saved record/sidebar association and read-only
  behavior. Existing save tests retain collision protection and read-back proof.
- `test_cloud_workflow_schema.py` replays the actual frontend consumer corpus,
  including URL acceptance, opaque object boundaries, no network/base and
  fail-closed dependency absence. Source provenance lives in
  [cloud_workflow_provenance.md](../../comfy_cli/schemas/cloud_workflow_provenance.md).
- Skill/reference tests and frozen agent scenarios distinguish per-file decisions,
  repair limits, prepared state and simulated verification from actual jobs.
- Run `python scripts/check_adrs.py`, the repository's full pytest suite, Ruff
  lint and formatting checks. A production rollout additionally needs the matching
  browser/backend release and capability checks; an unavailable capability stays
  disabled rather than falling back to guessed links.

## Revisit triggers

Reopen this decision if any changed source byte reaches model upload despite a
matching expected digest; if a public-only match is reported as owned reuse; if
any private link selects a different record/workspace/release; if frontend/Python
acceptance differs on one oracle case; or if a supported Python/OS target lacks a
usable Ada installation. A request for atomic guarantees on another upload
mechanism requires a separately reviewed protocol decision.
