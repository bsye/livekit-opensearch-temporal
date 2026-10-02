import { llm, voice } from '@livekit/agents'
import type { ActionDecision } from '@voice/temporal'
import { z } from 'zod'

interface Approval {
  approved: boolean
  reason?: string
}

export interface ApprovalToolOptions<S extends z.AnyZodObject> {
  name: string
  description: string
  parameters: S
  needsApproval: boolean | ((args: z.infer<S>) => boolean)
  describe: (args: z.infer<S>) => string
  execute: (args: z.infer<S>) => Promise<string>
  report: (decision: ActionDecision) => void
  participant: () => string
}

export function approvalTool<S extends z.AnyZodObject>(opts: ApprovalToolOptions<S>) {
  return llm.tool({
    description: opts.description,
    parameters: opts.parameters,
    execute: async (args: z.infer<S>, { ctx }) => {
      const need = typeof opts.needsApproval === 'function' ? opts.needsApproval(args) : opts.needsApproval
      if (need) {
        const started = Date.now()
        const record = (decision: ActionDecision['decision'], reasons: string[] = []) =>
          opts.report({
            tool: opts.name,
            args,
            decision,
            reasons,
            latencyMs: Date.now() - started,
            at: Date.now(),
            participant: opts.participant(),
          })
        const what = opts.describe(args)
        record('confirm')
        const history = ctx.session.chatCtx.copy({ excludeInstructions: true, excludeFunctionCall: true })
        const result = await ctx.foreground(() => confirmTask(what, history).run())
        record(result.approved ? 'confirmed' : 'declined', result.reason ? [result.reason] : [])
        if (!result.approved) {
          return (
            `Not done: the user did not approve "${what}"${result.reason ? ` (${result.reason})` : ''}. ` +
            `If they corrected something, call ${opts.name} again with the corrected values; otherwise ask what they want.`
          )
        }
      }
      return opts.execute(args)
    },
  })
}

function confirmTask(what: string, chatCtx: llm.ChatContext): voice.AgentTask<Approval> {
  const task: voice.AgentTask<Approval> = voice.AgentTask.create<Approval>({
    chatCtx,
    instructions:
      `You are confirming a single action with the user before it happens: ${what}. ` +
      `If they clearly agree, call approve. If they decline, hesitate, or change any detail, call reject ` +
      `with their reason or correction. Don't discuss anything else; keep it to one short sentence.`,
    tools: {
      approve: llm.tool({
        description: 'The user clearly agreed to the action.',
        execute: async () => {
          task.complete({ approved: true })
          return 'Approved.'
        },
      }),
      reject: llm.tool({
        description: 'The user declined, was unsure, or changed a detail.',
        parameters: z.object({ reason: z.string().describe("The user's reason or correction, in their words") }),
        execute: async ({ reason }) => {
          task.complete({ approved: false, reason })
          return 'Rejected.'
        },
      }),
    },
    onEnter: ({ session }) => {
      session.say(`Just to confirm: ${what}?`)
    },
  })
  return task
}
