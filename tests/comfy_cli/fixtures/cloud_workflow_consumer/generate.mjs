import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = process.env.FRONTEND_ROOT
if (!root) throw new Error('FRONTEND_ROOT must identify the authoritative frontend checkout')
const schemaPath = 'src/platform/workflow/validation/schemas/workflowSchema.ts'
const { validateComfyWorkflow } = await import(pathToFileURL(join(root, schemaPath)).href)
const node = () => ({ id: 1, type: 'Note', pos: [0, 0], size: [200, 100], flags: {}, order: 0, mode: 0, properties: {} })
const state = () => ({ lastGroupId: 0, lastNodeId: 1, lastLinkId: 1, lastRerouteId: 0 })
const graph = (version) => version === 1
  ? { version: 1, state: state(), nodes: [node()], links: [] }
  : { version, last_node_id: 1, last_link_id: 1, nodes: [node()], links: [] }
const subgraph = () => ({ ...graph(1), id: 'c327cae8-2acf-4bfa-bb88-97948a3c008b', revision: 1, name: 'Nested', inputNode: { id: -10, bounding: [0, 0, 100, 100] }, outputNode: { id: -20, bounding: [0, 0, 100, 100] } })
const cases = []
const add = (id, input) => cases.push({ id, input })
const mutate = (version, label, edit) => {
  const input = graph(version)
  edit(input)
  add(`${version}/${label}`, input)
}
const slotValues = ['0', '1', '-1', '+12', '  3  ', '\t4', '12tail', '1.9', '0x10', '-0xF', '0b10', '0o10', '1e3', '1_000', '', ' ', 'junk', 'NaN', 'Infinity', '0x', '9007199254740992', '9'.repeat(400), '\u00a012tail', '\ufeff-0x10', '\u008512', '\u200b1', '-0', '+0xF', '0Xff', '\u202812', '١', '12١', 0, -1, 0.5, 9007199254740992, null, true]
const urlVectors = [
  ['empty',''], ['relative-path','models/foo'], ['relative-root','/models/foo'], ['relative-parent','../foo'], ['relative-query','?model=foo'], ['relative-fragment','#foo'], ['relative-protocol','//example.com/foo'],
  ['https-no-host','https://'], ['https-missing-host-path','https:///foo'], ['https-one-slash','https:/foo'], ['https-no-slash','https:foo'], ['https-empty-port','https://example.com:/foo'], ['https-negative-port','https://example.com:-1/foo'], ['https-max-port','https://example.com:65535/foo'], ['https-overflow-port','https://example.com:65536/foo'],
  ['file-empty','file:'], ['file-local','file:///tmp/model'], ['file-host','file://localhost/tmp/model'], ['file-relative','file:relative'], ['file-drive','file:///C:/model'],
  ['opaque-custom','custom:opaque'], ['opaque-empty','custom:'], ['custom-host','custom://example.com/path'], ['custom-no-host','custom:///path'], ['opaque-space','custom:hello world'], ['opaque-unicode','custom:é🙂'], ['data','data:text/plain,model'], ['about','about:blank'], ['mailto-empty','mailto:'], ['javascript-empty','javascript:'], ['invalid-scheme','1custom:path'], ['scheme-space','custom :path'],
  ['percent-lone','https://example.com/%'], ['percent-short','https://example.com/%2'], ['percent-invalid','https://example.com/%GG'], ['percent-host-invalid','https://exa%mple.com/'], ['percent-host-invalid-hex','https://%GG/'], ['percent-host-unicode','https://%C3%A9xample.com/'], ['percent-host-bad-utf8','https://%FF/'], ['percent-encoded-surrogate','https://example.com/%ED%A0%80'], ['percent-null-path','https://example.com/%00'], ['percent-null-host','https://%00/'],
  ['idn-unicode','https://例え.テスト/model'], ['idn-punycode','https://xn--r8jz45g.xn--zckzah/model'], ['idn-fullwidth','https://ｅｘａｍｐｌｅ.com/'], ['idn-bad-punycode','https://xn--/'], ['idn-joiner','https://a\u200db.com/'],
  ['ipv6-loopback','https://[::1]/model'], ['ipv6-full','https://[2001:db8::1]:443/model'], ['ipv6-unclosed','https://[::1/model'], ['ipv6-zone','https://[fe80::1%25en0]/'], ['ipv6-unbracketed','https://::1/model'], ['ipv4-hex','https://0x7f000001/model'], ['ipv4-short','https://127.1/model'], ['ipv4-bad','https://256.1.1.1/model'],
  ['credentials','https://user:pass@example.com/model'], ['backslashes','https:\\example.com\model'], ['space-host','https://exa mple.com/model'], ['newline-scheme','ht\ntps://example.com/model'], ['tab-scheme','ht\ttps://example.com/model']
]
for (const [label, scalar] of [['high-surrogate','\ud800'],['low-surrogate','\udfff'],['pair','\ud83d\ude42'],['replacement','\ufffd'],['reversed-pair','\udfff\ud800']]) {
  urlVectors.push([`path-${label}`,`https://example.com/${scalar}`], [`query-${label}`,`https://example.com/?q=${scalar}`], [`fragment-${label}`,`https://example.com/#${scalar}`], [`host-${label}`,`https://${scalar}.example/`], [`opaque-${label}`,`custom:${scalar}`])
}
for (const code of [...Array.from({length:33},(_,i)=>i),127]) {
  const scalar=String.fromCharCode(code)
  const label=code.toString(16).padStart(2,'0')
  urlVectors.push([`c0-${label}-leading`,`${scalar}https://example.com/model`], [`c0-${label}-trailing`,`https://example.com/model${scalar}`], [`c0-${label}-path`,`https://example.com/a${scalar}b`], [`c0-${label}-host`,`https://exa${scalar}mple.com/`])
}
for (const version of [0.4, 1]) {
  add(`${version}/minimal`, graph(version))
  mutate(version, 'empty-canvas', g => { g.nodes = [] })
  mutate(version, 'incomplete-node', g => { g.nodes = [{}] })
  for (const field of ['id', 'type', 'pos', 'size', 'flags', 'order', 'mode', 'properties']) mutate(version, `missing-node-${field}`, g => { delete g.nodes[0][field] })
  for (const field of ['version', 'nodes', ...(version === 1 ? ['state'] : ['last_node_id', 'last_link_id', 'links'])]) mutate(version, `missing-graph-${field}`, g => { delete g[field] })
  for (const [index, slot] of slotValues.entries()) {
    mutate(version, `link-slot-${index}`, g => { g.links = version === 1 ? [{ id: 1, origin_id: 1, origin_slot: slot, target_id: 1, target_slot: slot, type: 'IMAGE' }] : [[1, 1, slot, 1, slot, 'IMAGE']] })
    mutate(version, `input-slot-${index}`, g => { g.nodes[0].inputs = [{ name: 'in', type: 'IMAGE', slot_index: slot }] })
  }
  for (const [index, tail] of ['\n', '\r', '\r\n', '\u2028', '\u2029'].entries()) {
    for (const [field,value] of [['cnr_id','abc'],['aux_id','foo/repo'],['ver','1.2.3']]) mutate(version, `metadata-line-ending-${field}-${index}`, g => { g.nodes[0].properties[field]=value+tail })
  }
  for (const [index, value] of ['valid-id', 'a', 'abc.def_12-foo', '', '_bad', 'bad-', 'éx', 'a/b', 'a'.repeat(101)].entries()) mutate(version, `cnr-${index}`, g => { g.nodes[0].properties.cnr_id = value })
  for (const [index, value] of ['foo/bar', 'Foo-1/repo.name', '-foo/repo', 'foo-/repo', 'foo--bar/repo', 'foo/_repo', 'foo/repo_', 'a'.repeat(40)+'/repo', 'foo/bar/baz', 'é/repo', 'a/b', 'a/r'.repeat(52)].entries()) mutate(version, `aux-${index}`, g => { g.nodes[0].properties.aux_id = value })
  for (const [index, value] of ['unknown', '1.2.3', 'v1.2.3', '1.2.3-beta.1+build.4', '1.2.3-01', '01.2.3', '1.2', 'bad', 'abcd', 'ABCD', 'vABCD', 'abc', 'a'.repeat(40), 'a'.repeat(41), 'V1.2.3', 'vunknown', '1.2.3-Beta', '1२.2.3', '', 1, null].entries()) mutate(version, `ver-${index}`, g => { g.nodes[0].properties.ver = value })
  for (const [index, value] of [[null, 1, true, 'x', {}, [1, {'x':'y'}]], { positive: 'hello', nested: { x: [] }, length: 10001 }, [], {}, 'text', null, 1, true].entries()) mutate(version, `widget-${index}`, g => { g.nodes[0].widgets_values = value })
  for (const [index, key] of ['0', '-1', '1', '01', '+1', '-0', '1.0', '1e3', ' 1', '', '9007199254740991', '9007199254740992'].entries()) mutate(version, `link-presentation-key-${index}`, g => { g.extra = { linkPresentation: { [key]: { hidden: true } } } })
  mutate(version, 'vector-object-normalization', g => { g.nodes[0].pos = {0: 7, 1: -2, extra: 'removed'}; g.nodes[0].size = {0: 200, 1: 100} })
  mutate(version, 'bad-vector-object', g => { g.nodes[0].pos = {0: 7} })
  mutate(version, 'arbitrary-property-kept', g => { g.nodes[0].properties.custom = { fn: 'opaque', list: [] } })
  mutate(version, 'recursive-subgraphs', g => { const sub = subgraph(); sub.definitions = { subgraphs: [subgraph()] }; g.definitions = {subgraphs:[sub]} })
  mutate(version, 'recursive-subgraph-invalid-slot', g => { const sub = subgraph(); const nested=subgraph(); nested.nodes[0].outputs=[{name:'x',type:'IMAGE',slot_index:'NaN'}]; sub.definitions={subgraphs:[nested]};g.definitions={subgraphs:[sub]} })
  mutate(version, 'recursive-subgraph-invalid-pack', g => { const sub = subgraph(); const nested=subgraph(); nested.nodes[0].properties.ver='nonsense'; sub.definitions={subgraphs:[nested]};g.definitions={subgraphs:[sub]} })
  mutate(version, 'recursive-subgraph-vector-normalization', g => { const sub = subgraph(); sub.nodes[0].pos={0:1,1:2}; g.definitions={subgraphs:[sub]} })
  mutate(version, 'recursive-subgraph-missing-state', g => { const sub = subgraph(); delete sub.state; g.definitions={subgraphs:[sub]} })
  mutate(version, 'root-unknown-field-kept', g => { g.customMetadata={x:1,ver:'invalid',origin_slot:'NaN',nodes:[{properties:{ver:'invalid'}}]} })
  mutate(version, 'opaque-widget-metadata-kept', g => { g.nodes[0].widgets_values=[{ver:'invalid',origin_slot:'NaN',nodes:[{}]}] })
  for (const [label,url] of urlVectors) mutate(version, `url-boundary-${label}`, g => { g.models=[{name:'test.safetensors',url,directory:'checkpoints'}] })
  for (const [index,url] of ['https://example.com/model.safetensors','ftp://example.com/file','mailto:a@example.com','not-a-url','https://','javascript:alert(1)','https://a b.com/file','https://example.com/a b',' https://example.com/model ','https://éxample.com/file','https://example.com:99999/file','https:example.com/file'].entries()) mutate(version, `model-url-${index}`, g => {g.models=[{name:'test.safetensors',url,directory:'checkpoints'}]})
}
add('root/not-an-object', [])
add('root/no-version', {nodes:[]})
add('root/string-version', {...graph(0.4),version:'0.4'})
add('root/unknown-numeric-version-uses-04', graph(99))
for (const test of cases) {
  const normalized=await validateComfyWorkflow(test.input,()=>{})
  test.accepted=normalized!==null
  if (normalized!==null) test.normalized=normalized
}
const sources=[schemaPath,'src/types/linkId.ts','src/lib/litegraph/src/types/widgets.ts']
const provenance={ frontend_commit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(), oracle:'validateComfyWorkflow(input, () => {})', node_version:process.versions.node, zod_version:JSON.parse(readFileSync(join(root,'node_modules/zod/package.json'),'utf8')).version, frontend_lockfile_sha256:createHash('sha256').update(readFileSync(join(root,'pnpm-lock.yaml'))).digest('hex'), generated_at:'2026-10-24', sources:Object.fromEntries(sources.map(p=>[p,createHash('sha256').update(readFileSync(join(root,p))).digest('hex')])), export_command:'FRONTEND_ROOT=<frontend-checkout> node --import tsx <canonical-cli>/tests/comfy_cli/fixtures/cloud_workflow_consumer/generate.mjs (cwd frontend)' }
const destination=join(dirname(fileURLToPath(import.meta.url)),'corpus.json')
const json=JSON.stringify({provenance,nonfinite_normalized_encoding:'Infinity/-Infinity/NaN emitted as {$number: <string>} only in normalized output; inputs are plain JSON',cases},(key,value)=>typeof value==='number'&&!Number.isFinite(value)?{$number:String(value)}:value,2)
writeFileSync(destination,json+'\n')
console.log(destination,`${cases.length} cases`,`${cases.filter(c=>c.accepted).length} accepted`)
