import { sleep } from '@temporalio/workflow'
import type { EmailInput, ReminderInput } from '../types.js'

export async function reminder(input: ReminderInput): Promise<ReminderInput & { firedAt: number }> {
  const wait = input.fireAt - Date.now()
  if (wait > 0) await sleep(wait, { summary: `⏰ until ${input.when}` })
  return { ...input, firedAt: Date.now() }
}

export async function sendEmail(input: EmailInput): Promise<EmailInput & { sentAt: number; simulated: true }> {
  return { ...input, sentAt: Date.now(), simulated: true }
}
