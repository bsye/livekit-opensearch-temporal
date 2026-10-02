/**
 * Index the turns of past room workflows into conversation memory. Idempotent.
 *   npm run backfill-memory -w @voice/worker [-- --skip-users alice,tester]
 */
import { indexExchange } from '@voice/memory'
import { connectTemporal, type RoomState } from '@voice/temporal'

const skipArg = process.argv.indexOf('--skip-users')
const skipUsers = new Set(skipArg > 0 ? process.argv[skipArg + 1].split(',') : [])

const client = await connectTemporal()
let rooms = 0
let indexed = 0
for await (const wf of client.workflow.list({ query: 'WorkflowType="roomSession"' })) {
  const handle = client.workflow.getHandle(wf.workflowId, wf.runId)
  let state: RoomState
  try {
    state = wf.status.name === 'RUNNING' ? await handle.query<RoomState>('roomState') : await handle.result()
  } catch {
    continue // failed or terminated runs have no result
  }
  const users = Object.values(state.participants)
    .filter((p) => p.kind !== 'AGENT')
    .map((p) => p.identity)
  const turns = (state.turns ?? []).filter((t) => t.userText && t.reply)
  if (users.some((u) => skipUsers.has(u)) || !turns.length) continue
  rooms++
  for (const t of turns) {
    await indexExchange({
      roomSid: state.sid ?? wf.workflowId,
      roomName: state.name,
      user: t.user,
      agent: t.agent,
      userText: t.userText,
      agentText: t.reply,
      startedAt: t.startedAt,
      endedAt: t.endedAt,
      fromMemory: t.fromMemory,
    })
    indexed++
  }
  console.log(`${state.name ?? wf.workflowId}: ${turns.length} turns`)
}
console.log(`indexed ${indexed} exchanges from ${rooms} rooms`)
