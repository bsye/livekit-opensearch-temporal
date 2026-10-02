import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { expandQuery } from '@voice/memory';

/** The product's query expansion, cached per question so reruns don't call the LLM again. */
export class QueryExpander {
  private cache: Record<string, string>;

  constructor(private cacheFile: URL) {
    this.cache = existsSync(cacheFile) ? (JSON.parse(readFileSync(cacheFile, 'utf8')) as Record<string, string>) : {};
  }

  async expand(questionId: string, question: string): Promise<string> {
    if (this.cache[questionId] === undefined) {
      this.cache[questionId] = await expandQuery(question);
      writeFileSync(this.cacheFile, JSON.stringify(this.cache));
    }
    return this.cache[questionId];
  }
}
