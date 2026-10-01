// Temporal client helpers shared by the translator and the agents (not usable in workflows).
import { Client, Connection, WorkflowExecutionAlreadyStartedError } from '@temporalio/client';
import { WorkflowIdReusePolicy } from '@temporalio/common';
import { RoomName, signalLabel, TASK_QUEUE, type EmailInput, type ReminderInput, type RoomSignal } from './shared.js';
import { reminder, roomSession, sendEmail } from './workflows.js';

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

/** Start a durable reminder (set_reminder tool); listed under the room's RoomName. */
export async function startReminder(client: Client, input: ReminderInput, roomName?: string): Promise<string> {
  const handle = await client.workflow.start(reminder, {
    workflowId: `reminder-${input.roomSid}-${input.fireAt}`,
    taskQueue: TASK_QUEUE,
    args: [input],
    staticSummary: `⏰ ${input.text} @ ${input.when}`,
    typedSearchAttributes: roomName ? [{ key: RoomName, value: roomName }] : [],
  });
  return handle.workflowId;
}

/** Undo for set_reminder: cancels the reminder's timer workflow. */
export async function cancelReminder(client: Client, workflowId: string): Promise<void> {
  await client.workflow.getHandle(workflowId).cancel();
}

/** Record an approved (simulated) email as a workflow, listed under the room's RoomName. */
export async function recordEmail(client: Client, input: EmailInput, roomName?: string): Promise<string> {
  const handle = await client.workflow.start(sendEmail, {
    workflowId: `email-${input.roomSid}-${Date.now()}`,
    taskQueue: TASK_QUEUE,
    args: [input],
    staticSummary: `✉️ to ${input.to}: ${input.subject}`,
    typedSearchAttributes: roomName ? [{ key: RoomName, value: roomName }] : [],
  });
  await handle.result();
  return handle.workflowId;
}
