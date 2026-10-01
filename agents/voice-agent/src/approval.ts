// Tool approval, modelled on the OpenAI Agents SDK's `needsApproval` and built from LiveKit
// primitives: a tool that needs approval pauses inside its own execute(), runs a small
// confirmation AgentTask in the foreground (ctx.foreground), and only continues if the user
// approves. The LLM can't skip it or claim success early: the tool is still running until
// the task returns.
//
// Risk tiers: reversible actions (reminders) run immediately, read back what they did and have
// an undo tool; consequential actions (sending, paying, deleting) declare needsApproval.
// Laya isn't in the path: it audits committed actions afterwards (audit.ts).
import { llm, voice } from '@livekit/agents';
import type { GateDecision } from 'livekit-temporal/shared';
import { z } from 'zod';

interface Approval {
  approved: boolean;
  reason?: string; // e.g. the user's correction when they reject
}

export interface ApprovalToolOptions<S extends z.AnyZodObject> {
  name: string;
  description: string;
  parameters: S;
  /** true = always ask; a function decides per call from the arguments (e.g. amount > 100). */
  needsApproval: boolean | ((args: z.infer<S>) => boolean);
  /** The action as read back to the user: "send the email to Marco". */
  describe: (args: z.infer<S>) => string;
  execute: (args: z.infer<S>) => Promise<string>;
  /** Called for each approval step, for Temporal's timeline. */
  report: (decision: GateDecision) => void;
  participant: () => string;
}

export function approvalTool<S extends z.AnyZodObject>(opts: ApprovalToolOptions<S>) {
  return llm.tool({
    description: opts.description,
    parameters: opts.parameters,
    execute: async (args: z.infer<S>, { ctx }) => {
      const need = typeof opts.needsApproval === 'function' ? opts.needsApproval(args) : opts.needsApproval;
      if (need) {
        const started = Date.now();
        const record = (decision: GateDecision['decision'], reasons: string[] = []) =>
          opts.report({
            tool: opts.name,
            args,
            decision,
            reasons,
            latencyMs: Date.now() - started,
            at: Date.now(),
            participant: opts.participant(),
          });
        const what = opts.describe(args);
        record('confirm');
        // The task sees the conversation so far (not the system prompt or this pending tool call),
        // so it understands replies like "no, tell him Thursday"
        const history = ctx.session.chatCtx.copy({ excludeInstructions: true, excludeFunctionCall: true });
        const result = await ctx.foreground(() => confirmTask(what, history).run());
        record(result.approved ? 'confirmed' : 'declined', result.reason ? [result.reason] : []);
        if (!result.approved) {
          return (
            `Not done: the user did not approve "${what}"${result.reason ? ` (${result.reason})` : ''}. ` +
            `If they corrected something, call ${opts.name} again with the corrected values; otherwise ask what they want.`
          );
        }
      }
      return opts.execute(args);
    },
  });
}

/** Takes over the conversation until the user approves or rejects one action. */
function confirmTask(what: string, chatCtx: llm.ChatContext): voice.AgentTask<Approval> {
  const question = `Just to confirm: ${what}?`;
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
          task.complete({ approved: true });
          return 'Approved.';
        },
      }),
      reject: llm.tool({
        description: 'The user declined, was unsure, or changed a detail.',
        parameters: z.object({ reason: z.string().describe("The user's reason or correction, in their words") }),
        execute: async ({ reason }) => {
          task.complete({ approved: false, reason });
          return 'Rejected.';
        },
      }),
    },
    // Spoken verbatim, no LLM call: faster, and the wording can't drift
    onEnter: ({ session }) => {
      session.say(question);
    },
  });
  return task;
}
