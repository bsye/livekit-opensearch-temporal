/**
 * Index the transcripts of past room workflows into conversation memory. Idempotent.
 *   npm run backfill-memory -w @voice/worker [-- --skip-users alice,tester]
 */
import { indexExchange } from '@voice/memory';
import { connectTemporal, type RoomState, type TranscriptEntry } from '@voice/temporal';

const skipArg = process.argv.indexOf('--skip-users');
const skipUsers = new Set(skipArg > 0 ? process.argv[skipArg + 1].split(',') : []);

const client = await connectTemporal();
let rooms = 0;
let indexed = 0;
for await (const wf of client.workflow.list({ query: 'WorkflowType="roomSession"' })) {
  const handle = client.workflow.getHandle(wf.workflowId, wf.runId);
  let state: RoomState;
  try {
    state = wf.status.name === 'RUNNING' ? await handle.query<RoomState>('roomState') : await handle.result();
  } catch {
    continue; // failed or terminated runs have no result
  }
  const users = Object.values(state.participants)
    .filter((p) => p.kind !== 'AGENT')
    .map((p) => p.identity);
  if (users.some((u) => skipUsers.has(u)) || !state.transcript?.length) continue;
  rooms++;
  for (const exchange of exchanges(state.transcript)) {
    await indexExchange({ roomSid: state.sid ?? wf.workflowId, roomName: state.name, ...exchange });
    indexed++;
  }
  console.log(`${state.name ?? wf.workflowId}: ${state.transcript.length} transcript lines`);
}
console.log(`indexed ${indexed} exchanges from ${rooms} rooms`);

function* exchanges(transcript: TranscriptEntry[]) {
  let userTurns: TranscriptEntry[] = [];
  for (const entry of transcript) {
    if (entry.role !== 'assistant') {
      userTurns.push(entry);
      continue;
    }
    if (userTurns.length === 0) continue;
    yield {
      user: userTurns[0].participant,
      agent: entry.participant,
      userText: userTurns.map((t) => t.text).join(' '),
      agentText: entry.text,
      startedAt: userTurns[0].at,
      endedAt: entry.at,
      fromMemory: entry.fromMemory,
    };
    userTurns = [];
  }
}
