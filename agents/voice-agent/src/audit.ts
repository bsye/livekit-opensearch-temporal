// Laya as auditor: after an action has run, score it against what the user said, off the
// critical path. Low scores are flagged in Temporal for review; over time the flagged and
// unflagged actions become labelled data to fine-tune Laya on this domain.
import type { GateDecision } from 'livekit-temporal/shared';

const FLAG_BELOW = 0.5;

export class ActionAuditor {
  constructor(
    private layaBaseUrl: string,
    private participant: () => string,
    private recentUserText: () => string,
    private report: (decision: GateDecision) => void,
  ) {}

  /** Fire-and-forget: never delays the conversation, never throws. */
  audit(tool: string, args: Record<string, unknown>): void {
    void this.score(tool, args).catch((err) => console.error('audit failed', err));
  }

  private async score(tool: string, args: Record<string, unknown>): Promise<void> {
    const started = Date.now();
    const call = `${tool}(${Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`;
    const res = await fetch(`${this.layaBaseUrl}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        state: `User request: ${this.recentUserText()}\nAction taken: ${call}`,
        questions: {
          match: { type: 'noul', instructions: 'Does the action taken do what the user asked, with the same details?' },
        },
      }),
    });
    if (!res.ok) throw new Error(`laya ${res.status}`);
    const intent = Number(((await res.json()) as { answers: { match: { noul: number } } }).answers.match.noul);
    this.report({
      tool,
      args,
      decision: intent < FLAG_BELOW ? 'flagged' : 'audited',
      reasons: intent < FLAG_BELOW ? [`audit ${intent.toFixed(2)}: may not match the request`] : [],
      intent,
      latencyMs: Date.now() - started,
      at: Date.now(),
      participant: this.participant(),
    });
  }
}
