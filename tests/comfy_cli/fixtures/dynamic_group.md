# DynamicGroup fixture

The schema was captured from a real ComfyUI `/object_info` response using core
`2d2fa46e18d293ced1b958fe288730e2ef4b322e` and the frontend devtool node at
`c2ba6f397ec7fa0bd5d5b93562a1b9d141fc37c9`.

The populated case contains the serialized DynamicGroup node and its matching
frontend prompt inputs captured during browser execution. The node was extracted
from its subgraph into a single-node workflow; its output link was removed. Widget
values, named values, and input definitions were retained. The execution returned
the expected A/C rows, although that browser scenario separately failed an outer
subgraph UI cleanup assertion.

The empty case is reduced from the same frontend revision's
`browser_tests/assets/inputs/dynamic_group.json`. Its expected inputs omit the
zero-row controller, as required by the prompt contract.
