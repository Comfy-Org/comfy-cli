from comfy_cli.command.custom_nodes.bisect_custom_nodes import parse_cm_output

CM_OUTPUT_REAL = """\
FETCH ComfyRegistry Data: 5/85
FETCH ComfyRegistry Data: 10/85
FETCH ComfyRegistry Data: 15/85
FETCH ComfyRegistry Data: 20/85
FETCH ComfyRegistry Data: 25/85
FETCH ComfyRegistry Data: 30/85
FETCH ComfyRegistry Data: 35/85
FETCH ComfyRegistry Data: 40/85
FETCH ComfyRegistry Data: 45/85
FETCH ComfyRegistry Data: 50/85
FETCH ComfyRegistry Data: 55/85
FETCH ComfyRegistry Data: 60/85
FETCH ComfyRegistry Data: 65/85
FETCH ComfyRegistry Data: 70/85
FETCH ComfyRegistry Data: 75/85
FETCH ComfyRegistry Data: 80/85
FETCH ComfyRegistry Data: 85/85
FETCH ComfyRegistry Data [DONE]
FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json [DONE]
FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json [DONE]
A3D ComfyUI Integration@1.0.2
ComfyUI_ACE-Step@1.1.2
Bjornulf_custom_nodes@1.1.1
cg-use-everywhere@6.1.0
ComfyUI-0246@1.1.3
ComfyUI ArtVenture@nightly
comfyui-auto-nodes-layout@0.0.1
ComfyUI-ConDelta@nightly
ComfyUI-Custom-Scripts@nightly
ComfyUI-DiaTTS@0.3.0
ComfyUI-Easy-Use@1.3.0
ComfyUI F5-TTS@nightly
ComfyUI-Florence2@1.0.3
ComfyUI@nightly
ComfyUI-GGUF@nightly
ComfyUI-Image-Filters@nightly
ComfyUI-KJNodes@nightly
comfyui-lmstudio-image-to-text-node@1.1.14
ComfyUI-LogicUtils@1.7.2
ComfyUI-LTXVideo@nightly
ComfyUI-Manager@nightly
ComfyUI-MMAudio@1.0.2
ComfyUI-mxToolkit@0.9.92
ComfyUI-VideoHelperSuite@1.6.1
ComfyUI Web Viewer@1.0.32
comfyui_controlnet_aux@1.0.7
ComfyUI_IPAdapter_plus@2.0.0
Prompt Stash@1.2.0
efficiency-nodes-comfyui@1.0.6
gguf@2.1.0
comfyui_HiDream-Sampler@1.0.0
LF Nodes@0.7.0
lora-info@1.0.2
Masquerade Nodes@nightly
rgthree-comfy@nightly
ComfyUI-TeaCache@1.5.1
WAS Node Suite@1.0.2
ComfyUI-ultimate-openpose-editor@nightly
ComfyUI-Dia@unknown
ComfyUI-Orpheus@unknown
"""

EXPECTED_NODES = [
    "A3D ComfyUI Integration@1.0.2",
    "ComfyUI_ACE-Step@1.1.2",
    "Bjornulf_custom_nodes@1.1.1",
    "cg-use-everywhere@6.1.0",
    "ComfyUI-0246@1.1.3",
    "ComfyUI ArtVenture@nightly",
    "comfyui-auto-nodes-layout@0.0.1",
    "ComfyUI-ConDelta@nightly",
    "ComfyUI-Custom-Scripts@nightly",
    "ComfyUI-DiaTTS@0.3.0",
    "ComfyUI-Easy-Use@1.3.0",
    "ComfyUI F5-TTS@nightly",
    "ComfyUI-Florence2@1.0.3",
    "ComfyUI@nightly",
    "ComfyUI-GGUF@nightly",
    "ComfyUI-Image-Filters@nightly",
    "ComfyUI-KJNodes@nightly",
    "comfyui-lmstudio-image-to-text-node@1.1.14",
    "ComfyUI-LogicUtils@1.7.2",
    "ComfyUI-LTXVideo@nightly",
    "ComfyUI-Manager@nightly",
    "ComfyUI-MMAudio@1.0.2",
    "ComfyUI-mxToolkit@0.9.92",
    "ComfyUI-VideoHelperSuite@1.6.1",
    "ComfyUI Web Viewer@1.0.32",
    "comfyui_controlnet_aux@1.0.7",
    "ComfyUI_IPAdapter_plus@2.0.0",
    "Prompt Stash@1.2.0",
    "efficiency-nodes-comfyui@1.0.6",
    "gguf@2.1.0",
    "comfyui_HiDream-Sampler@1.0.0",
    "LF Nodes@0.7.0",
    "lora-info@1.0.2",
    "Masquerade Nodes@nightly",
    "rgthree-comfy@nightly",
    "ComfyUI-TeaCache@1.5.1",
    "WAS Node Suite@1.0.2",
    "ComfyUI-ultimate-openpose-editor@nightly",
    "ComfyUI-Dia@unknown",
    "ComfyUI-Orpheus@unknown",
]


class TestParseCmOutput:
    def test_real_output_filters_fetch_lines(self):
        result = parse_cm_output(CM_OUTPUT_REAL)
        assert result == EXPECTED_NODES
        assert len(result) == 40

    def test_no_fetch_lines_in_result(self):
        result = parse_cm_output(CM_OUTPUT_REAL)
        for node in result:
            assert not node.startswith("FETCH"), f"FETCH line leaked: {node}"

    def test_pinned_nodes_excluded(self):
        pinned = {"ComfyUI-Manager@nightly", "ComfyUI@nightly"}
        result = parse_cm_output(CM_OUTPUT_REAL, pinned)
        assert "ComfyUI-Manager@nightly" not in result
        assert "ComfyUI@nightly" not in result
        assert len(result) == 38

    def test_empty_output(self):
        assert parse_cm_output("") == []
        assert parse_cm_output("   \n  \n  ") == []

    def test_only_fetch_lines(self):
        output = "FETCH ComfyRegistry Data: 5/85\nFETCH DATA from: foo [DONE]\n"
        assert parse_cm_output(output) == []

    def test_no_fetch_lines(self):
        output = "NodeA@1.0\nNodeB@nightly\n"
        assert parse_cm_output(output) == ["NodeA@1.0", "NodeB@nightly"]

    def test_arbitrary_status_lines_filtered(self):
        output = "some random status line\nINFO: loading\nNodeA@1.0\nDone.\n"
        assert parse_cm_output(output) == ["NodeA@1.0"]
