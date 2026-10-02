import { ask } from '@voice/laya'
import type { ActionDecision } from '@voice/temporal'

const FLAG_BELOW = 0.5

/**
 * Laya scores each committed action against what the user asked, off the critical path. Flags are
 * for review (and labelled data to fine-tune Laya later), never a block: zero-shot it isn't reliable enough.
 */
export class ActionAuditor {
  constructor(
    private participant: () => string,
    private recentUserText: () => string,
    private report: (decision: ActionDecision) => void,
  ) {}

  audit(tool: string, args: Record<string, unknown>): void {
    this.score(tool, args).catch((err) => console.error('audit failed', err))
  }

  private async score(tool: string, args: Record<string, unknown>): Promise<void> {
    const started = Date.now()
    const call = `${tool}(${Object.entries(args)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(', ')})`
    const { answers } = await ask(`User request: ${this.recentUserText()}\nAction taken: ${call}`, {
      match: { type: 'noul', instructions: 'Does the action taken do what the user asked, with the same details?' },
    })
    const intent = answers.match.noul ?? 0
    const flagged = intent < FLAG_BELOW
    this.report({
      tool,
      args,
      decision: flagged ? 'flagged' : 'audited',
      reasons: flagged ? [`audit ${intent.toFixed(2)}: may not match the request`] : [],
      intent,
      latencyMs: Date.now() - started,
      at: Date.now(),
      participant: this.participant(),
    })
  }
}
