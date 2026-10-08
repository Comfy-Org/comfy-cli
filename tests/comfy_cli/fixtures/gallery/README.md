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
for f in 02_qwen_Image_edit_subgraphed 05_audio_ace_step_1_t2a_song_subgraphed image_z_image_turbo audio_minimax_music_3 api_seedance2_5_video_extend templates_graphic_design_recomposer template_horizontal_vertical_extension flux_dev_checkpoint_example; do
  curl -sfL "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/${f}.json" -o "${f}.json"
done
```

`qwen_object_info.json` is NOT a template — it is a purpose-built minimal
`/object_info` covering only the node types the qwen fixture uses (frontend-only
`MarkdownNote` excluded, matching a real server response), used to exercise slot
extraction. If a refresh changes the qwen template's node set, extend it to match.

`image_z_image_turbo.json` (pre-migration save: proxies backed by linked
inputs, no host values), `audio_minimax_music_3.json` (post-migration save:
host values differ from the interior defaults) and
`api_seedance2_5_video_extend.json` (socket + widget inputs mixed) back the
promoted-widget tests (`tests/comfy_cli/cql/test_promoted_inputs.py`,
`tests/comfy_cli/command/test_workflow_edit_promoted.py`,
`tests/comfy_cli/test_workflow_to_api_promoted.py`); their node classes are
covered by `../object_info_subgraph_promoted.json`, a trimmed copy of the
cloud catalog entries (tooltips stripped, structure verbatim).

`templates_graphic_design_recomposer.json` (two `PrimitiveNode` fan-outs and a
labelled value widget promoted only through legacy `proxyWidgets`),
`template_horizontal_vertical_extension.json` (three `$$canvas-image-preview`
exposures plus a value widget whose interior node serializes no `inputs[]`)
and `flux_dev_checkpoint_example.json` (seven linked promotions, one
repairable `seed`, one unrepairable `control_after_generate`) back the
legacy-proxy repair tests (`tests/comfy_cli/cql/test_proxy_migration.py`,
`tests/comfy_cli/command/test_workflow_edit_legacy_proxy.py`) — the port of the
frontend's `proxyWidgetMigration.ts`. Their node classes are covered by
`../object_info_legacy_proxy.json`, built the same way as
`object_info_subgraph_promoted.json`: for every class the three templates use,
the cloud catalog entry (`services/ingest/data/object_info.json`) verbatim with
every `tooltip` key stripped; frontend-only classes (`MarkdownNote`, `Note`,
`PrimitiveNode`) have no catalog entry and are omitted.
