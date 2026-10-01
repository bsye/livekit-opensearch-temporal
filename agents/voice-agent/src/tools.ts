// The agent's tools, by risk tier:
//   reversible    set_reminder (runs immediately, reads back) + cancel_reminder (the undo)
//   consequential send_email (needsApproval: the user confirms before it runs)
// Every committed action is audited by Laya afterwards (audit.ts).
// recall searches conversation memory (BM25 + MiniLM re-ranking, Laya routing; no embeddings).
import { llm } from '@livekit/agents';
import type { Client } from '@temporalio/client';
import { cancelReminder, recordEmail, startReminder } from 'livekit-temporal/client';
import { recall } from 'livekit-temporal/memory';
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
  onRecall: () => void; // marks the next agent reply as answered from memory
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

  const recallTool = llm.tool({
    description:
      'Search past conversations with the user, e.g. "what did we say about the Lisbon trip?" or ' +
      '"what did I ask you last week?". Returns matching past messages with when they happened.',
    parameters: z.object({
      question: z.string().describe("The user's question, in their own words"),
      from: z.string().optional().describe('Start of the time range, ISO date/time, only if the user gave one'),
      to: z.string().optional().describe('End of the time range, ISO date/time, only if the user gave one'),
    }),
    execute: async ({ question, from, to }) => {
      deps.onRecall();
      const { hits, route, timings, ms } = await recall({
        question,
        from: from ? Date.parse(from) || undefined : undefined,
        to: to ? Date.parse(to) || undefined : undefined,
        excludeRoomSid: deps.room().sid,
      });
      console.log(`recall [${route}] "${question}": ${hits.length} hits in ${ms}ms ${JSON.stringify(timings)}`);
      if (hits.length === 0) return 'Nothing found about that in past conversations.';
      // chronological, so facts that changed over time read in order
      const lines = [...hits]
        .sort((a, b) => a.doc.endedAt - b.doc.endedAt)
        .map(({ doc }) => `[${formatWhen(doc.endedAt)}] User: ${doc.userText} | Assistant: ${doc.agentText.slice(0, 300)}`);
      return (
        `Relevant past messages, oldest first:\n${lines.join('\n')}\n` +
        'Answer from these, saying when it was. If the user said different things at different times ' +
        '(a number, a plan, a preference changed), the most recent statement is the current one. ' +
        'If none of them answers the question, say you do not remember.'
      );
    },
  });

  return {
    set_reminder: setReminder,
    cancel_reminder: cancelReminderTool,
    send_email: sendEmail,
    recall: recallTool,
  };
}

/** "Tue 1 Oct, 13:05" in local time, for reading back when something was said. */
function formatWhen(ms: number): string {
  const d = new Date(ms);
  return `${d.toDateString().slice(0, 10)}, ${d.toTimeString().slice(0, 5)}`;
}

/** Next occurrence of HH:MM local time today or tomorrow, as unix ms. */
function fireTime(time: string, day: 'today' | 'tomorrow'): number {
  const [h, m] = time.split(':').map(Number);
  const at = new Date();
  at.setHours(h, m, 0, 0);
  if (day === 'tomorrow') at.setDate(at.getDate() + 1);
  return at.getTime();
}
