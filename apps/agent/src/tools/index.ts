import { llm } from '@livekit/agents'
import { NOTHING_FOUND, recall } from '@voice/memory'
import type { ActionDecision, RoomRef } from '@voice/temporal'
import { z } from 'zod'
import type { Actions } from './actions.js'
import { approvalTool } from './approval.js'
import type { RecallPrefetch } from './prefetch.js'

export interface ToolDeps {
  room: () => RoomRef
  agent: () => string
  reportAction: (decision: ActionDecision) => void
  /** Marks the next reply as answered from memory. */
  onRecall: () => void
  /** Filled with each call's duration, by tool call id. */
  toolTimings?: Map<string, number>
  /** A search already started on the user's words (see prefetch.ts). */
  prefetch?: RecallPrefetch
}

/**
 * The cascade's LLM tools, by risk: reminders run at once and can be undone; sending an email
 * needs the user's approval first. Laya audits every action afterwards.
 */
export function createTools(actions: Actions, deps: ToolDeps) {
  const tools = {
    set_reminder: llm.tool({
      description:
        'Set a reminder for the user at a time today or tomorrow. Only when the user asks to be reminded. ' +
        'It is set immediately; tell the user what was set so they can correct it.',
      parameters: z.object({
        text: z.string().describe('What to remind about, short, e.g. "call mom"'),
        time: z.string().describe('24-hour time HH:MM, e.g. "18:00" for 6pm'),
        day: z.enum(['today', 'tomorrow']),
      }),
      execute: (args) => actions.setReminder(args),
    }),

    cancel_reminder: llm.tool({
      description:
        'Cancel a reminder set in this conversation, e.g. when the user says "undo", "cancel that" or corrects it ' +
        '(then cancel and set a new one). Without text, cancels the most recent one.',
      parameters: z.object({
        text: z.string().optional().describe('Part of the reminder text, to pick which one'),
      }),
      execute: (args) => actions.cancelReminder(args),
    }),

    send_email: approvalTool({
      name: 'send_email',
      description: 'Send an email for the user (simulated). The user is asked to approve it before it is sent.',
      parameters: z.object({
        to: z.string().describe('Recipient name or address'),
        subject: z.string(),
        body: z.string(),
      }),
      needsApproval: true,
      describe: ({ to, subject }) => `send the email to ${to} about "${subject}"`,
      execute: (args) => actions.sendEmail(args),
      report: deps.reportAction,
      participant: deps.agent,
    }),

    recall: llm.tool({
      description:
        'Search past conversations with the user, e.g. "what did we say about the Lisbon trip?" or ' +
        '"what did I ask you last week?". Returns matching past messages with when they happened.',
      parameters: z.object({
        question: z.string().describe("The user's question, in their own words"),
        from: z.string().optional().describe('Start of the time range, ISO date/time, only if the user gave one'),
        to: z.string().optional().describe('End of the time range, ISO date/time, only if the user gave one'),
      }),
      execute: async ({ question, from, to }) => {
        deps.onRecall()
        const started = Date.now()
        // the prefetched search used the user's own words; a time range, or nothing found there
        // (a follow-up the model rephrased), needs a search on the model's question
        const early = !from && !to ? deps.prefetch?.current() : undefined
        let result = early ? await early.result.catch(() => undefined) : undefined
        const prefetched = !!result?.hits.length
        if (!prefetched) {
          result = await recall({
            question,
            from: from ? Date.parse(from) || undefined : undefined,
            to: to ? Date.parse(to) || undefined : undefined,
            excludeRoomSid: deps.room().sid,
          })
        }
        const { hits, route, timings } = result!
        console.log(
          `recall [${route}, ${prefetched ? `prefetched "${early!.question}"` : 'searched'}] "${question}": ${hits.length} hits, ${Date.now() - started}ms on the critical path ${JSON.stringify(timings)}`,
        )
        if (hits.length === 0) return NOTHING_FOUND
        const lines = hits
          .toSorted((a, b) => a.doc.endedAt - b.doc.endedAt)
          .map(
            ({ doc }) =>
              `[${formatWhen(doc.endedAt)}] User: ${doc.userText} | Assistant: ${doc.agentText.slice(0, 300)}`,
          )
        return (
          `Relevant past messages, oldest first:\n${lines.join('\n')}\n` +
          'Answer from these, saying when it was. If the user said different things at different times ' +
          '(a number, a plan, a preference changed), the most recent statement is the current one. ' +
          'If none of them answers the question, say you do not remember.'
        )
      },
    }),
  }
  if (deps.toolTimings) for (const tool of Object.values(tools)) timed(tool, deps.toolTimings)
  return tools
}

/** Records how long each call took (approval included: it runs inside execute). */
function timed(tool: { execute: (...args: never[]) => Promise<unknown> }, timings: Map<string, number>) {
  const execute = tool.execute.bind(tool) as (args: unknown, opts: { toolCallId: string }) => Promise<unknown>
  tool.execute = (async (args: unknown, opts: { toolCallId: string }) => {
    const started = Date.now()
    try {
      return await execute(args, opts)
    } finally {
      timings.set(opts.toolCallId, Date.now() - started)
    }
  }) as never
}

function formatWhen(ms: number): string {
  const d = new Date(ms)
  return `${d.toDateString().slice(0, 10)}, ${d.toTimeString().slice(0, 5)}`
}
