// Run from the matching frontend checkout with its locked dependencies:
// FRONTEND_ROOT="$PWD" node --import tsx /path/to/cli/scripts/export_cloud_workflow_schemas.mjs /output/schema-dir
import fs from 'node:fs/promises'
import path from 'node:path'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

const frontend = process.env.FRONTEND_ROOT
if (!frontend || !process.argv[2]) throw new Error('FRONTEND_ROOT and output directory are required')
const require = createRequire(path.join(frontend, 'package.json'))
const { z } = require('zod')
const source = await import(pathToFileURL(path.join(frontend, 'src/platform/workflow/validation/schemas/workflowSchema.ts')))
await fs.mkdir(process.argv[2], { recursive: true })
for (const [version, schema] of [['0_4', source.zComfyWorkflow], ['1_0', source.zComfyWorkflow1]]) {
  const generated = z.toJSONSchema(schema, { io: 'input', unrepresentable: 'any' })
  await fs.writeFile(path.join(process.argv[2], `cloud_workflow_${version}.json`), JSON.stringify(generated, null, 2))
}
