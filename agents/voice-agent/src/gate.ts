// Action gate: decides whether a tool call can run straight away or needs the user's spoken
// confirmation first. Three layers, each doing what it is reliable at:
//   1. intent  — Laya (local decision model, ~5ms): is this the kind of action the user asked for?
//   2. values  — plain code: every value in the call must match what the user actually said
//   3. confirm — when 1 or 2 flag it (or the tool is high-stakes), ask the user; Laya reads the reply
// Laya alone approved "50 euros" → amount=500 at p=1.00, hence layer 2 (services/laya/eval_gate.py).
import type { GateDecision } from 'livekit-temporal/shared';

const INTENT_THRESHOLD = 0.5;
// Measured: "yes"-type replies 0.84–0.98, "no"/"not sure" 0.10–0.14, "wait, actually, not now" 0.61
const AGREE_THRESHOLD = 0.75;
const DECLINE_THRESHOLD = 0.25;

export interface ProposedAction {
  tool: string;
  args: Record<string, unknown>;
  /** How the agent reads the action back to the user, e.g. "a reminder to call mom at 18:00 today". */
  description: string;
  /** Layer 2 for this tool: reasons the values don't match what the user said (empty = fine). */
  checkValues: (userText: string) => string[];
  /** Always confirm, regardless of layers 1–2 (money, deletions, …). */
  highStakes?: boolean;
}

export type GateOutcome =
  | { run: true; decision: GateDecision }
  | { run: false; decision: GateDecision; tellModel: string };

export class ActionGate {
  private pending?: { key: string; question: string; askedAt: number };

  constructor(
    private layaBaseUrl: string,
    private participant: () => string,
    private recentUserText: (since?: number) => string,
    private report: (decision: GateDecision) => void,
  ) {}

  async check(action: ProposedAction): Promise<GateOutcome> {
    const started = Date.now();
    const key = `${action.tool}:${JSON.stringify(action.args)}`;
    const decide = (decision: GateDecision['decision'], reasons: string[], extra: Partial<GateDecision> = {}) => {
      const d: GateDecision = {
        tool: action.tool,
        args: action.args,
        decision,
        reasons,
        latencyMs: Date.now() - started,
        at: Date.now(),
        participant: this.participant(),
        ...extra,
      };
      this.report(d);
      return d;
    };

    // Second call for an action we asked about: did the user agree since we asked?
    if (this.pending?.key === key) {
      const reply = this.recentUserText(this.pending.askedAt);
      if (reply) {
        const agreement = await this.agreement(this.pending.question, reply);
        if (agreement >= AGREE_THRESHOLD) {
          this.pending = undefined;
          return { run: true, decision: decide('confirmed', [], { agreement }) };
        }
        if (agreement <= DECLINE_THRESHOLD) {
          this.pending = undefined;
          return {
            run: false,
            decision: decide('declined', [], { agreement }),
            tellModel: `The user did not confirm, so nothing was done. Ask what they would like instead.`,
          };
        }
        return this.askToConfirm(action, key, decide('confirm', ['unclear answer'], { agreement }));
      }
    }

    const userText = this.recentUserText();
    const reasons = action.checkValues(userText);
    const intent = await this.intent(userText, action);
    if (intent < INTENT_THRESHOLD) reasons.unshift(`intent ${intent.toFixed(2)}`);
    if (action.highStakes) reasons.push('high-stakes action');

    if (reasons.length === 0) return { run: true, decision: decide('pass', [], { intent }) };
    return this.askToConfirm(action, key, decide('confirm', reasons, { intent }));
  }

  private askToConfirm(action: ProposedAction, key: string, decision: GateDecision): GateOutcome {
    const question = `Just to confirm: ${action.description}, right?`;
    this.pending = { key, question, askedAt: Date.now() };
    return {
      run: false,
      decision,
      tellModel:
        `Not done yet: this needs the user's confirmation. Ask exactly: "${question}" ` +
        `If they agree, call ${action.tool} again with the same arguments. If they correct something, call it with the corrected arguments.`,
    };
  }

  /** Layer 1: P(the proposed action is what the user asked for). */
  private async intent(userText: string, action: ProposedAction): Promise<number> {
    const call = `${action.tool}(${Object.entries(action.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`;
    const answers = await this.ask(`User request: ${userText}\nProposed action: ${call}`, {
      type: 'noul',
      instructions: 'Does the proposed action do exactly what the user asked, with the same details?',
    });
    return Number(answers.noul ?? 0);
  }

  /** Layer 3: P(the user's reply agrees to the confirmation question). Neutral keys: Laya's noul follows its labels. */
  private async agreement(question: string, reply: string): Promise<number> {
    const answers = await this.ask(`Assistant asked: ${question}\nUser replied: ${reply}`, {
      type: 'choice',
      instructions: 'How did the user reply to the confirmation question?',
      criteria: {
        A: 'The user agreed: yes, go ahead.',
        B: 'The user declined, corrected it, or asked for something else.',
      },
    });
    return Number((answers.probabilities as Record<string, number> | undefined)?.A ?? 0);
  }

  private async ask(state: string, question: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.layaBaseUrl}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ state, questions: { q: question } }),
    });
    if (!res.ok) throw new Error(`laya ${res.status}: ${await res.text()}`);
    const body = (await res.json()) as { answers: Record<string, Record<string, unknown>> };
    return body.answers.q;
  }
}

// --- Layer 2 helpers -----------------------------------------------------------------------

const NUMBER_WORDS: Record<string, number> = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, noon: 12, midnight: 0,
};
const PM_WORDS = /\b(pm|p\.m\.|evening|tonight|afternoon|night)\b/i;
const AM_WORDS = /\b(am|a\.m\.|morning)\b/i;

/** Hours the user mentioned ("6", "six", "18:00", "6pm"), as written. */
function hoursMentioned(text: string): number[] {
  const hours: number[] = [];
  for (const m of text.matchAll(/\b(\d{1,2})(?::\d{2})?\s*(?:o'?clock)?/gi)) hours.push(Number(m[1]));
  for (const [word, n] of Object.entries(NUMBER_WORDS)) if (new RegExp(`\\b${word}\\b`, 'i').test(text)) hours.push(n);
  return hours.filter((h) => h >= 0 && h <= 23);
}

/** Layer 2 for a time like "18:00" against what the user said. */
export function checkTime(time: string, day: string, userText: string): string[] {
  const reasons: string[] = [];
  const hour = Number(time.split(':')[0]);
  const mentioned = hoursMentioned(userText);
  if (mentioned.length === 0) reasons.push('no time in the request');
  else if (!mentioned.some((h) => h === hour || h + 12 === hour || (h === 12 && hour === 0))) {
    reasons.push(`time ${time} not in the request`);
  }
  if (PM_WORDS.test(userText) && hour < 12) reasons.push(`${time} is morning, the user said evening`);
  if (AM_WORDS.test(userText) && hour >= 12) reasons.push(`${time} is afternoon/evening, the user said morning`);
  const saidTomorrow = /\btomorrow\b/i.test(userText);
  if (saidTomorrow && day !== 'tomorrow') reasons.push(`day ${day}, the user said tomorrow`);
  if (!saidTomorrow && day === 'tomorrow' && /\b(today|tonight)\b/i.test(userText)) reasons.push('day tomorrow, the user said today');
  return reasons;
}
