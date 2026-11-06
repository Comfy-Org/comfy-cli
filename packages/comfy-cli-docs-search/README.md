# comfy-cli-docs-search

This companion package is generated and published alongside comfy-cli. It
contains the version-matched LanceDB index, precomputed documentation vectors,
and the local ONNX query encoder used by `comfy docs search --mode hybrid`.

Install the matching optional extra with `comfy-cli[docs-search]`. The runtime
does not download models or index data. Generated assets are produced by
`scripts/build_docs_lancedb.py` during the release build and are not checked in
as source files.

The bundled encoder is the FastEmbed-compatible BGE small English ONNX export from
`qdrant/bge-small-en-v1.5-onnx-q` at the pinned revision recorded in the pack
manifest. That model is MIT licensed; the accompanying model notice must be
included in the generated pack.
