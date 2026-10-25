# Open a custom environment in Comfy Cloud

`comfy --json deploy ui --deployment <id>` discovers an authenticated editor link without opening a browser, selecting a deployment, uploading a file or starting a job. It accepts the same deployment name/id and optional spec path as `deploy show`.

A successful result contains `workspaceId`, `deploymentId`, `releaseId` and `url`. The URL has the established Cloud editor shape:

```
https://cloud.comfy.org/?workspace=<workspace-id>&deployment=<deployment-id>
```

The command checks both the ready, nondeleted deployment control record and the authenticated Cloud workspace's editor deployment listing. The exact deployment and served release must agree. The Cloud listing owns rollout eligibility, current membership and workspace access. A ready deployment API endpoint alone does not establish editor access.

A Cloud sign-in is required. API keys can run deployment API jobs, but do not establish a browser session or permission to select an editor environment. Refused, missing or inconsistent capability data returns `deploy_ui_unavailable` without a guessed URL. Opening a successful link still requires the browser to sign in as an account with access; no credential is embedded in the URL.

The frontend switches to the named workspace, asks the backend to select the exact deployment for that browser and reloads onto the selected release's node catalog. The backend rechecks access at selection and execution time. Discovery is a point-in-time check; a deployment can change or be deleted before a link is opened.

The returned link targets an environment; opening it selects that environment in the browser. A standard-environment workflow sidebar save does not select its custom environment. A specific saved workflow needs its own authenticated workflow-opening contract; `#<workflow-id>` is not that contract. Where advertised, `comfy workflow ui` verifies that contract and the normal saved document before returning a private link.

## Contract owners

- Platform: `composables/useDeploymentEditorLink.ts` (`deploymentEditorUrl`, readiness and server capability checks).
- ComfyUI frontend: `src/platform/workspace/composables/useDeploymentUrlLoader.ts` (workspace switch, authenticated pick and reload).
- Cloud ingest: `GET /api/workspaces/{id}/deployments` (`ListWorkspaceDeployments`), `PUT /api/workspaces/{id}/deployment` (`PickWorkspaceDeployment`), the selected release catalog and prompt delegation.

These source contracts describe available implementation. They do not prove that a particular backend/frontend revision or rollout is deployed. The CLI obtains the server's current answer on each discovery.
