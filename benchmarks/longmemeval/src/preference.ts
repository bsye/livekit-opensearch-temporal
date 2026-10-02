/**
 * Laya preference tags (tested, not adopted): turns stating a preference are boosted for questions Laya
 * judges to ask for one. Gold question types are never used. Scores are cached per distinct text.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { scan } from '@voice/laya';

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

  async turnScores(texts: string[]): Promise<number[]> {
    const missing = [...new Set(texts.filter((t) => this.scores[hash(t)] === undefined))];
    for (let s = 0; s < missing.length; s += SCAN_CHUNK) {
      const chunk = missing.slice(s, s + SCAN_CHUNK);
      const { scores } = await scan(chunk, TURN_QUESTION);
      chunk.forEach((t, i) => (this.scores[hash(t)] = scores[i]));
      this.dirty += chunk.length;
    }
    if (this.dirty > 20000) this.save();
    return texts.map((t) => this.scores[hash(t)]);
  }

  async queryScore(question: string): Promise<number> {
    return (await scan([`User question: ${question}`], QUERY_QUESTION)).scores[0];
  }

  save(): void {
    writeFileSync(this.cacheFile, JSON.stringify(this.scores));
    this.dirty = 0;
  }
}

