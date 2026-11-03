# Saved-canvas schema provenance

These files are generated from the matching frontend consumer, not a handwritten subset.

Frontend base commit: `78492d6a93837f192a4ae21582cdf05ef8bcb9ad`. The source hashes also capture uncommitted changes.

- `src/platform/workflow/validation/schemas/workflowSchema.ts` SHA256 `9485268f5cd12305ffb03d4ffc711588846a59fb603bb4311add0a16063e4a49`
- `src/types/linkId.ts` SHA256 `ebcec86190f31e0e3a5738659cdf00d8b07850388b7ac7d663cbe9beaf0546e0`
- `src/lib/litegraph/src/types/widgets.ts` SHA256 `31af22f25acb97b868b7006b9b6ca9fb4f5d401608019040f774fdde8243ca85`

Generated structural schema files (byte-identical input exports):

- `cloud_workflow_0_4.json` SHA256 `6b0b93c0b37ae24947723f98e99e13b9d53ced9ef45ff6d83dc0c83d4decbf5f`
- `cloud_workflow_1_0.json` SHA256 `e76e9948865b34b14fe6616b4a37a7932143bcde225eb6b6d544117c6d371377`

Regenerate from the frontend checkout with its locked dependencies:

```sh
FRONTEND_ROOT="$PWD" node --import tsx /path/to/cli/scripts/export_cloud_workflow_schemas.mjs /path/to/cli/comfy_cli/schemas
```

The export uses `z.toJSONSchema(schema, {io: "input", unrepresentable: "any"})`, matching the frontend generator. `cloud_workflow_schema.py` annotates only known schema positions to restore the omitted input acceptance checks: slot parseInt validity, aux_id repository rules, pack version rules, canonical presentation link IDs, and a strict cnr_id fullmatch to bridge Python/JavaScript final-newline regex semantics. It never interprets opaque widget/metadata object keys as graph structure. Vector normalization changes frontend output, not acceptance; widget custom validation accepts every plain JSON value.

`tests/comfy_cli/fixtures/cloud_workflow_consumer/corpus.json` has 826 inputs accepted/rejected by actual `validateComfyWorkflow`; the generator records normalized outputs. Regenerate that corpus after every schema source change and compare Python acceptance before releasing a private-link capability.

URL input acceptance uses the WHATWG Ada parser through `ada-url>=4,<5` (verified with 4.0.0). The Python adapter converts strings to Unicode scalar values using UTF-16 surrogate replacement, matching browser USVString handling, and calls `URL.can_parse` without a base or network access. It validates acceptance only and preserves the original graph bytes. The official 4.0.0 wheels require Python 3.10 or newer and cover the project's existing CPython Windows, macOS, and Linux CI targets; other architectures/interpreters may require the source distribution. No Node runtime is required by the CLI.

All 826 frontend oracle inputs match Python acceptance, including 466 URL cases, with three additional adapter tests for network independence, absent implicit base, and fail-closed parser unavailability. The complete 829-test group also passes independently on Python 3.10.21 and 3.12.11. These results apply to the recorded frontend source and corpus; repeat the comparison when either changes.
