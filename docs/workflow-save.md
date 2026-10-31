# Saving a workflow to the Cloud sidebar

`comfy --json workflow save workflow.json --name "My workflow" --where cloud`
writes the normal `workflows/My workflow.json` userdata file once. It returns
`filename` and, when supported by the server, `saved_file_id`: the actual userdata
UUID from that write receipt. The UUID survives ordinary saves and renames; a
deleted and recreated file has a new identity.

`workflow_id` remains the separate mirrored workflow-record ID used by the
record-based list/get/delete commands. A same-name record with equal content does
not establish the saved file's UUID. Never substitute `workflow_id` for
`saved_file_id` or construct a private editor URL from either value alone.

Older servers can save successfully without returning a usable file UUID. In
that case `saved_file_id` is null and `saved_file_identity_unavailable` explains
that private file-link certification is unavailable. Verify the saved file in
Cloud's Workflows sidebar; do not repeat the save or any local file uploads to
obtain an ID. File identity metadata does not grant upload permission.
