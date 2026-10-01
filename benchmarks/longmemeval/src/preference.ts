// Laya preference tags: at index time Laya scores whether each turn states the user's own preference;
// at query time whether the question asks for something tailored to the user's preferences. Turns
// that state preferences are boosted for such questions. The dataset's gold question types are never
// used. Scores are cached per distinct text (LongMemEval_M repeats each turn ~4x).
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';
export const TURN_QUESTION = 'Does the user state their own preference, taste, interest, like or dislike?';
export const QUERY_QUESTION =
  'Is the user asking for a recommendation or suggestion that should fit their own preferences or interests?';
const SCAN_CHUNK = 1000;

const hash = (text: string) => createHash('sha1').update(text).digest('base64').slice(0, 16);

export class PreferenceTags {
  private scores: Record<string, number>;
  private dirty = 0;

  constructor(private cacheFile: URL) {
    this.scores = existsSync(cacheFile) ? (JSON.parse(readFileSync(cacheFile, 'utf8')) as Record<string, number>) : {};
  }

  /** P(turn states a preference) for each text; Laya is called only for texts not seen before. */
  async turnScores(texts: string[]): Promise<number[]> {
    const missing = [...new Set(texts.filter((t) => this.scores[hash(t)] === undefined))];
    for (let s = 0; s < missing.length; s += SCAN_CHUNK) {
      const chunk = missing.slice(s, s + SCAN_CHUNK);
      const scores = await scan(chunk, TURN_QUESTION);
      chunk.forEach((t, i) => (this.scores[hash(t)] = scores[i]));
      this.dirty += chunk.length;
    }
    if (this.dirty > 20000) this.save();
    return texts.map((t) => this.scores[hash(t)]);
  }

  /** P(question asks for something tailored to the user's preferences). */
  async queryScore(question: string): Promise<number> {
    return (await scan([`User question: ${question}`], QUERY_QUESTION))[0];
  }

  save(): void {
    writeFileSync(this.cacheFile, JSON.stringify(this.scores));
    this.dirty = 0;
  }
}

async function scan(states: string[], question: string): Promise<number[]> {
  const res = await fetch(`${LAYA}/v1/scan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ states, question }),
  });
  if (!res.ok) throw new Error(`laya scan: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { scores: number[] }).scores;
}
