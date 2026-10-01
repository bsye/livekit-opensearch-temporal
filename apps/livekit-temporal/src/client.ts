// Temporal client helpers shared by the translator and the agents (not usable in workflows).
import { Client, Connection, WorkflowExecutionAlreadyStartedError } from '@temporalio/client';
import { WorkflowIdReusePolicy } from '@temporalio/common';
import { RoomName, signalLabel, TASK_QUEUE, type RoomSignal } from './shared.js';
import { roomSession } from './workflows.js';

export async function connectTemporal(): Promise<Client> {
  return new Client({
    connection: await Connection.connect({ address: process.env.TEMPORAL_ADDRESS ?? 'localhost:7233' }),
    namespace: process.env.TEMPORAL_NAMESPACE ?? 'default',
  });
}

/**
 * Signal the RoomSession workflow for a room session, starting it if needed
 * (webhooks and agents race, so whoever arrives first starts it). The signal is named after
 * what happened and who did it (signalLabel), so the Temporal UI timeline is readable.
 * Returns false when the session already closed and the signal was dropped.
 */
export async function signalRoom(
  client: Client,
  room: { sid: string; name?: string },
  signal: RoomSignal,
): Promise<boolean> {
  try {
    await client.workflow.signalWithStart(roomSession, {
      workflowId: room.sid, // room names get reused, sids are unique per session
      taskQueue: TASK_QUEUE,
      signal: signalLabel(signal),
      signalArgs: [signal],
      // a closed session must not be restarted by a late signal
      workflowIdReusePolicy: WorkflowIdReusePolicy.REJECT_DUPLICATE,
      typedSearchAttributes: room.name ? [{ key: RoomName, value: room.name }] : [],
      staticSummary: `🏠 ${room.name ?? room.sid}`, // label in the Temporal UI
    });
    return true;
  } catch (err) {
    if (err instanceof WorkflowExecutionAlreadyStartedError) return false;
    throw err;
  }
}
