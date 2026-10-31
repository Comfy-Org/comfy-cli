# Frontend saved-canvas acceptance oracle

`corpus.json` contains 826 JSON inputs, with verdicts and normalized outputs produced by actual frontend `validateComfyWorkflow`. No Python implementation supplied expected results. The checked-in generator documents/repeats the oracle call; provenance identifies frontend commit, Node26.8.2, Zod4.6.5, frontend lockfile SHA256 and SHA256s of all runtime sources used. No authentication, remote APIs or workflow execution is involved.

Run from the frontend checkout with its locked dependencies and supported Node:

```sh
FRONTEND_ROOT="$PWD" node --import tsx /absolute/canonical-cli/tests/comfy_cli/fixtures/cloud_workflow_consumer/generate.mjs
```

Each case contains `id`, `input`, `accepted`, and (when accepted) `normalized`. Nonfinite numbers in normalized output use `{"$number":"Infinity"}` tags; inputs remain plain JSON. The validator can produce Infinity from a long slot string, despite rejecting a nonfinite JSON number. Corpus covers both 0.4 and 1, node/graph required fields, recursive subgraph definitions, numeric string slots, model URLs, registry metadata, widgets, presentation ID keys, preserved opaque metadata and vector normalization.

## Acceptance semantics omitted by the generated JSON Schema

Use source-aware keywords on known schema positions. Do not recursively interpret arbitrary fields by name: opaque widget objects and unknown graph metadata can legitimately contain `ver`, `origin_slot`, or `nodes` keys.

- Slot string conversion is JavaScript `parseInt(value)` with no radix, accepted whenever the result is not NaN. Trim ECMAScript whitespace, optional sign, infer hexadecimal only for `0x`/`0X`, then consume the longest valid digit prefix. Examples: `12tail`→12, `1.9`→1, `1e3`→1, `0b10`→0, `-0xF`→-15. Empty/whitespace-only, `NaN`, `Infinity` and lone `0x` reject. String-derived unsafe integers and Infinity are accepted; number-derived slots must be safe integers. NBSP and BOM are whitespace, U+0085 and U+200B are not.
- `cnr_id` additional refine rejects start/end `[_\-.]`; its already-exported regex is stronger and contains this condition: `^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?$`, length1–100. Python JSON Schema uses Python regex semantics: its `$` can match before final LF, unlike the frontend regex. Therefore apply explicit full-match validation even though the generated pattern looks complete.
- `aux_id` splits exactly one slash and validates username length1–39 with `^(?!-)(?!.*--)[a-zA-Z0-9-]+(?<!-)$`, repo length1–100 with the same cnr regex. Normalization joins the same two segments without other changes.
- `properties.ver` strips one lowercase leading `v`, then validates semver/hash; a separate branch accepts the original value exactly `unknown`; the transformed branch accepts the following semver or hash regex. **Do not make `vunknown` accepted**: its transformed value is validated only by semver/hash. Hash: `^[0-9a-f]{4,40}$` case-insensitive. Semver: `^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([\da-z-]+(?:\.[\da-z-]+)*))?(?:\+([\da-z-]+(?:\.[\da-z-]+)*))?$` (lowercase identifiers only; prerelease numeric leading zeros accepted by this exact regex).
- `widgets_values` allows array or record. Each value may be any JSON value, including nested arbitrary objects/arrays. `isWidgetValue` accepts null/string/number/boolean/object, so no additional restriction is lost for plain JSON. Clipboard normalization is a separate API and does not run here.
- `extra.linkPresentation` keys require `Number.isSafeInteger(Number(key)) && String(Number(key)) === key`: canonical decimal safe integers, including negatives; reject `-0`, `01`, `+1`, `1.0`, `1e3`, whitespace, out-of-safe-range.
- Vector object `{0:number,1:number,...}` converts to tuple `[value[0],value[1]]`; unknown object fields disappear. This changes normalized output without changing acceptance.

Generated-schema annotations must cover both top-level and `$defs/__schema0` (recursive subgraph) schema bodies:

- `properties/nodes/items/properties/{inputs,outputs}/items/properties/slot_index`
- `properties/subgraphs/items/properties/{inputs,outputs}/items/properties/slot_index`
- `properties/floatingLinks/items/properties/{origin_slot,target_slot}`
- 1.0 and recursive 1.0 `properties/links/items/properties/{origin_slot,target_slot}`
- 0.4 `properties/links/items/prefixItems/{2,4}` (positions1 and3 are node IDs, **not** parsed slots)
- recursive exported graph `properties/{inputs,outputs}/items/properties/slot_index`
- `properties/nodes/items/properties/properties/properties/{cnr_id,aux_id,ver}`
- `properties/extra/anyOf/0/properties/linkPresentation/propertyNames`

The generated JSON schema's URI format needs a checker; the frontend uses WHATWG URL parsing. Corpus includes ordinary HTTPS, invalid URLs and non-HTTP schemes to prevent a narrower guessed HTTP-only validator.

## URL acceptance boundary expansion

442 additional cases exercise221 URL strings against both schema versions. Every C0 code point U+0000–U+0020 and DEL occurs at leading, trailing, path and host positions. Additional vectors cover isolated UTF16 high/low surrogates, reversed surrogate pairs, a valid emoji pair and U+FFFD in path/query/fragment/host/opaque positions; malformed percent escapes and encoded bytes; IDN/punycode; IPv6 and alternate IPv4 notation; missing hosts; file URLs; opaque/custom schemes; and relative/base-less URLs. Expected verdicts come exclusively from frontend Zod on the pinned Node runtime.

Browser URL construction accepts lone surrogates in a path/query/fragment or opaque URL by WebIDL USVString replacement, and rejects the resulting replacement character in a special-scheme host. A Python parser wrapper must apply equivalent scalar conversion for parsing, including proper pairing of explicit surrogate pairs. UTF8 `errors="replace"` alone is not equivalent (Python emits `?`, whereas USVString emits U+FFFD). The original JSON input remains unchanged; this corpus records frontend normalization separately and does not prescribe rewriting saved workflow bytes.

Lone surrogates are escaped in the JSON artifact, so the file is valid UTF8 JSON. Preserve these code units when reading fixtures; do not sanitize the input before comparing against the oracle.
