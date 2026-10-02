import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { completeChat } from '@voice/http'

export interface RecallCall {
  queries: string[]
  topic?: string
}

const EXAMPLE_QUERIES = ['Lisbon flights price', 'plane tickets Portugal cost', 'booked flight Lisbon']

export class QueryWriter {
  private cache: Record<string, RecallCall | string[]>

  constructor(
    private cacheFile: string | URL,
    private llm: { baseUrl: string; model: string },
    private topics?: Record<string, string>,
  ) {
    this.cache = existsSync(cacheFile) ? JSON.parse(readFileSync(cacheFile, 'utf8')) : {}
  }

  async call(key: string, question: string): Promise<RecallCall> {
    if (this.cache[key] === undefined) {
      this.cache[key] = await this.write(question)
      writeFileSync(this.cacheFile, JSON.stringify(this.cache))
    }
    const cached = this.cache[key]
    return Array.isArray(cached) ? { queries: cached } : cached
  }

  async queries(key: string, question: string): Promise<string[]> {
    return (await this.call(key, question)).queries
  }

  private async write(question: string): Promise<RecallCall> {
    const message = await completeChat(this.llm.baseUrl, {
      model: this.llm.model,
      temperature: 0,
      max_tokens: 120,
      tools: [this.tool()],
      messages: [{ role: 'system', content: this.system() }, ...this.example(), { role: 'user', content: question }],
    })
    const args = message.tool_calls?.[0]?.function.arguments
    const parsed = (args ? JSON.parse(args) : {}) as { queries?: unknown; topic?: unknown }
    const queries = Array.isArray(parsed.queries) ? parsed.queries : []
    return {
      queries: queries.filter((q): q is string => typeof q === 'string' && q.trim() !== '').slice(0, 3),
      topic: typeof parsed.topic === 'string' && this.topics && parsed.topic in this.topics ? parsed.topic : undefined,
    }
  }

  private system(): string {
    return (
      "You find what the user said in past conversations. For the user's question, call recall with 1 to 3 short " +
      'keyword queries: the words the user most likely used back then, one query per different way they may have ' +
      `talked about it${this.topics ? ', and the topic only if it is clear' : ''}. Never guess the answer.`
    )
  }

  private tool() {
    const queries = {
      type: 'array',
      items: { type: 'string' },
      minItems: 1,
      maxItems: 3,
      description: '1 to 3 short keyword queries',
    }
    const topic = this.topics && {
      type: 'string',
      enum: Object.keys(this.topics),
      description: `Only if clear: ${JSON.stringify(this.topics)}`,
    }
    return {
      type: 'function',
      function: {
        name: 'recall',
        description: "Search the user's past conversations.",
        parameters: { type: 'object', properties: { queries, ...(topic && { topic }) }, required: ['queries'] },
      },
    }
  }

  private example() {
    const args = { queries: EXAMPLE_QUERIES, ...(this.topics && { topic: 'travel' }) }
    return [
      { role: 'user', content: 'How much did I spend on the flights for my Lisbon trip?' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 'example', type: 'function', function: { name: 'recall', arguments: JSON.stringify(args) } },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'example',
        content: '[Mar 2] User: I just booked my flights to Lisbon, 420 euros return.',
      },
      { role: 'assistant', content: 'You paid 420 euros for the return flights.' },
    ]
  }
}
