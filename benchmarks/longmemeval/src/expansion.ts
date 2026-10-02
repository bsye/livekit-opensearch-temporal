import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { expandQuery } from '@voice/memory'

export class QueryExpander {
  private cache: Record<string, string>

  constructor(
    private cacheFile: URL,
    private llm?: { baseUrl: string; model: string },
  ) {
    this.cache = existsSync(cacheFile) ? (JSON.parse(readFileSync(cacheFile, 'utf8')) as Record<string, string>) : {}
  }

  async expand(questionId: string, question: string): Promise<string> {
    if (this.cache[questionId] === undefined) {
      this.cache[questionId] = await expandQuery(question, this.llm)
      writeFileSync(this.cacheFile, JSON.stringify(this.cache))
    }
    return this.cache[questionId]
  }
}
