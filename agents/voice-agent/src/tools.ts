// The agent's tools, by risk tier:
//   reversible    set_reminder (runs immediately, reads back) + cancel_reminder (the undo)
//   consequential send_email (needsApproval: the user confirms before it runs)
// Every committed action is audited by Laya afterwards (audit.ts).
import { llm } from '@livekit/agents';
import type { Client } from '@temporalio/client';
import { cancelReminder, recordEmail, startReminder } from 'livekit-temporal/client';
import type { GateDecision } from 'livekit-temporal/shared';
import { z } from 'zod';
import { approvalTool } from './approval.js';
import type { ActionAuditor } from './audit.js';

export interface ToolDeps {
  temporal: Client;
  room: () => { sid: string; name: string };
  user: () => string; // the human participant's identity
  agent: () => string; // the agent's identity
  auditor: ActionAuditor;
  report: (decision: GateDecision) => void;
}

export function createTools(deps: ToolDeps) {
  // Reminders set in this session, newest last, for cancel_reminder
  const reminders: { workflowId: string; text: string; when: string }[] = [];

  const setReminder = llm.tool({
    description:
      'Set a reminder for the user at a time today or tomorrow. Only when the user asks to be reminded. ' +
      'It is set immediately; tell the user what was set so they can correct it.',
    parameters: z.object({
      text: z.string().describe('What to remind about, short, e.g. "call mom"'),
      time: z.string().describe('24-hour time HH:MM, e.g. "18:00" for 6pm'),
      day: z.enum(['today', 'tomorrow']),
    }),
    execute: async ({ text, time, day }) => {
      if (!/^([01]?\d|2[0-3]):[0-5]\d$/.test(time)) return 'time must be 24-hour HH:MM; ask the user for the time.';
      const { sid, name } = deps.room();
      const when = `${time} ${day}`;
      const workflowId = await startReminder(
        deps.temporal,
        { text, fireAt: fireTime(time, day), when, roomSid: sid, participant: deps.user() },
        name,
      );
      reminders.push({ workflowId, text, when });
      deps.auditor.audit('set_reminder', { text, time, day });
      return `Done: reminder to ${text} at ${when}. Read this back to the user.`;
    },
  });

  const cancelReminderTool = llm.tool({
    description:
      'Cancel a reminder set in this conversation, e.g. when the user says "undo", "cancel that" or corrects it ' +
      '(then cancel and set a new one). Without text, cancels the most recent one.',
    parameters: z.object({
      text: z.string().optional().describe('Part of the reminder text, to pick which one'),
    }),
    execute: async ({ text }) => {
      const match = [...reminders].reverse().find((r) => !text || r.text.toLowerCase().includes(text.toLowerCase()));
      if (!match) return 'There is no matching reminder from this conversation.';
      await cancelReminder(deps.temporal, match.workflowId);
      reminders.splice(reminders.indexOf(match), 1);
      deps.auditor.audit('cancel_reminder', { text: match.text, when: match.when });
      return `Cancelled the reminder to ${match.text} at ${match.when}.`;
    },
  });

  const sendEmail = approvalTool({
    name: 'send_email',
    description: 'Send an email for the user (simulated). The user is asked to approve it before it is sent.',
    parameters: z.object({
      to: z.string().describe('Recipient name or address'),
      subject: z.string(),
      body: z.string(),
    }),
    needsApproval: true,
    describe: ({ to, subject }) => `send the email to ${to} about "${subject}"`,
    execute: async ({ to, subject, body }) => {
      const { sid, name } = deps.room();
      await recordEmail(deps.temporal, { to, subject, body, roomSid: sid, requestedBy: deps.user() }, name);
      deps.auditor.audit('send_email', { to, subject, body });
      return `Sent the email to ${to} (simulated).`;
    },
    report: deps.report,
    participant: deps.agent,
  });

  return { set_reminder: setReminder, cancel_reminder: cancelReminderTool, send_email: sendEmail };
}

/** Next occurrence of HH:MM local time today or tomorrow, as unix ms. */
function fireTime(time: string, day: 'today' | 'tomorrow'): number {
  const [h, m] = time.split(':').map(Number);
  const at = new Date();
  at.setHours(h, m, 0, 0);
  if (day === 'tomorrow') at.setDate(at.getDate() + 1);
  return at.getTime();
}
