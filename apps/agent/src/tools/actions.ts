import type { Client } from '@temporalio/client'
import { cancelReminder, type RoomRef, recordEmail, startReminder } from '@voice/temporal'
import type { ActionAuditor } from './audit.js'

export interface ActionDeps {
  temporal: Client
  room: () => RoomRef
  user: () => string
  auditor: ActionAuditor
}

export interface ReminderArgs {
  text: string
  time: string
  day: 'today' | 'tomorrow'
}

export interface EmailArgs {
  to: string
  subject: string
  body: string
}

export function createActions({ temporal, room, user, auditor }: ActionDeps) {
  const reminders: { workflowId: string; text: string; when: string }[] = []

  return {
    async setReminder({ text, time, day }: ReminderArgs): Promise<string> {
      if (!/^([01]?\d|2[0-3]):[0-5]\d$/.test(time)) return 'time must be 24-hour HH:MM; ask the user for the time.'
      const { sid, name } = room()
      const when = `${time} ${day}`
      const input = { text, fireAt: fireTime(time, day), when, roomSid: sid, participant: user() }
      reminders.push({ workflowId: await startReminder(temporal, input, name), text, when })
      auditor.audit('set_reminder', { text, time, day })
      return `Done: reminder to ${text} at ${when}. Read this back to the user.`
    },

    async cancelReminder({ text }: { text?: string }): Promise<string> {
      const match = reminders.findLast((r) => !text || r.text.toLowerCase().includes(text.toLowerCase()))
      if (!match) return 'There is no matching reminder from this conversation.'
      await cancelReminder(temporal, match.workflowId)
      reminders.splice(reminders.indexOf(match), 1)
      auditor.audit('cancel_reminder', { text: match.text, when: match.when })
      return `Cancelled the reminder to ${match.text} at ${match.when}.`
    },

    async sendEmail({ to, subject, body }: EmailArgs): Promise<string> {
      const { sid, name } = room()
      await recordEmail(temporal, { to, subject, body, roomSid: sid, requestedBy: user() }, name)
      auditor.audit('send_email', { to, subject, body })
      return `Sent the email to ${to} (simulated).`
    },
  }
}

export type Actions = ReturnType<typeof createActions>

function fireTime(time: string, day: 'today' | 'tomorrow'): number {
  const [h, m] = time.split(':').map(Number)
  const at = new Date()
  at.setHours(h, m, 0, 0)
  if (day === 'tomorrow') at.setDate(at.getDate() + 1)
  return at.getTime()
}
