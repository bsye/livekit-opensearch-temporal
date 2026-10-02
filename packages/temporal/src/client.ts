import { Client, Connection, WorkflowExecutionAlreadyStartedError } from '@temporalio/client'
import { WorkflowIdReusePolicy } from '@temporalio/common'
import { env } from '@voice/config'
import { RoomName, signalLabel, TASK_QUEUE } from './definitions.js'
import type { EmailInput, ReminderInput, RoomRef, RoomSignal } from './types.js'
import { reminder, roomSession, sendEmail } from './workflows/index.js'

export async function connectTemporal(): Promise<Client> {
  return new Client({
    connection: await Connection.connect({ address: env('TEMPORAL_ADDRESS') }),
    namespace: env('TEMPORAL_NAMESPACE'),
  })
}

const roomAttributes = (roomName?: string) => (roomName ? [{ key: RoomName, value: roomName }] : [])

export async function signalRoom(client: Client, room: RoomRef, signal: RoomSignal): Promise<boolean> {
  try {
    await client.workflow.signalWithStart(roomSession, {
      workflowId: room.sid,
      taskQueue: TASK_QUEUE,
      signal: signalLabel(signal),
      signalArgs: [signal],
      workflowIdReusePolicy: WorkflowIdReusePolicy.REJECT_DUPLICATE,
      typedSearchAttributes: roomAttributes(room.name),
      staticSummary: `🏠 ${room.name ?? room.sid}`,
    })
    return true
  } catch (err) {
    if (err instanceof WorkflowExecutionAlreadyStartedError) return false
    throw err
  }
}

export async function startReminder(client: Client, input: ReminderInput, roomName?: string): Promise<string> {
  const handle = await client.workflow.start(reminder, {
    workflowId: `reminder-${input.roomSid}-${input.fireAt}`,
    taskQueue: TASK_QUEUE,
    args: [input],
    staticSummary: `⏰ ${input.text} @ ${input.when}`,
    typedSearchAttributes: roomAttributes(roomName),
  })
  return handle.workflowId
}

export async function cancelReminder(client: Client, workflowId: string): Promise<void> {
  await client.workflow.getHandle(workflowId).cancel()
}

export async function recordEmail(client: Client, input: EmailInput, roomName?: string): Promise<string> {
  const handle = await client.workflow.start(sendEmail, {
    workflowId: `email-${input.roomSid}-${Date.now()}`,
    taskQueue: TASK_QUEUE,
    args: [input],
    staticSummary: `✉️ to ${input.to}: ${input.subject}`,
    typedSearchAttributes: roomAttributes(roomName),
  })
  await handle.result()
  return handle.workflowId
}
