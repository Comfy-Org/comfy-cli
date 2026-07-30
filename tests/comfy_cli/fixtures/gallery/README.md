# Gallery template fixtures

Real subgraphed gallery templates, copied **verbatim** from
[Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)
`templates/`. They back `tests/comfy_cli/test_subgraph_gallery_templates.py`,
the regression sweep that proves `convert_ui_to_api` fully expands real
subgraph instances and `Graph.get_template_schema` surfaces their interior
slots.

Refresh with:

```bash
cd tests/comfy_cli/fixtures/gallery
for f in 02_qwen_Image_edit_subgraphed 05_audio_ace_step_1_t2a_song_subgraphed; do
  curl -sfL "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/${f}.json" -o "${f}.json"
done
```

`qwen_object_info.json` is NOT a template — it is a purpose-built minimal
`/object_info` covering only the node types the qwen fixture uses (frontend-only
`MarkdownNote` excluded, matching a real server response), used to exercise slot
extraction. If a refresh changes the qwen template's node set, extend it to match.
