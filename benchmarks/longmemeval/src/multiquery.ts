import { existsSync, readFileSync, writeFileSync } from 'node:fs'

/**
 * The model writes its own searches: given the question, it calls recall with 1 to 3 short keyword
 * queries for the different ways the user may have talked about it. Cached per question.
 */
const SYSTEM =
  "You find what the user said in past conversations. For the user's question, call recall with 1 to 3 short " +
  'keyword queries: the words the user most likely used back then, one query per different way they may have ' +
  'talked about it. Never guess the answer.'

const TOOL = {
  type: 'function',
  function: {
    name: 'recall',
    description: "Search the user's past conversations.",
    parameters: {
      type: 'object',
      properties: {
        queries: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          maxItems: 3,
          description: '1 to 3 short keyword queries',
        },
      },
      required: ['queries'],
    },
  },
}

const EXAMPLE = [
  { role: 'user', content: 'How much did I spend on the flights for my Lisbon trip?' },
  {
    role: 'assistant',
    content: '',
    tool_calls: [
      {
        id: 'example',
        type: 'function',
        function: {
          name: 'recall',
          arguments: JSON.stringify({
            queries: ['Lisbon flights price', 'plane tickets Portugal cost', 'booked flight Lisbon'],
          }),
        },
      },
    ],
  },
  {
    role: 'tool',
    tool_call_id: 'example',
    content: '[Mar 2] User: I just booked my flights to Lisbon, 420 euros return.',
  },
  { role: 'assistant', content: 'You paid 420 euros for the return flights.' },
]

export class QueryWriter {
  private cache: Record<string, string[]>

  constructor(
    private cacheFile: URL,
    private llm: { baseUrl: string; model: string },
  ) {
    this.cache = existsSync(cacheFile) ? (JSON.parse(readFileSync(cacheFile, 'utf8')) as Record<string, string[]>) : {}
  }

  async queries(questionId: string, question: string): Promise<string[]> {
    if (this.cache[questionId] === undefined) {
      this.cache[questionId] = await this.write(question)
      writeFileSync(this.cacheFile, JSON.stringify(this.cache))
    }
    return this.cache[questionId]
  }

  private async write(question: string): Promise<string[]> {
    const res = await fetch(`${this.llm.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: this.llm.model,
        temperature: 0,
        max_tokens: 120,
        tools: [TOOL],
        messages: [{ role: 'system', content: SYSTEM }, ...EXAMPLE, { role: 'user', content: question }],
      }),
    })
    if (!res.ok) throw new Error(`query writer: ${res.status} ${await res.text()}`)
    const message = (
      (await res.json()) as { choices: { message: { tool_calls?: { function: { arguments: string } }[] } }[] }
    ).choices[0].message
    const args = message.tool_calls?.[0]?.function.arguments
    const queries = args ? ((JSON.parse(args) as { queries?: unknown }).queries ?? []) : []
    return Array.isArray(queries)
      ? queries.filter((q): q is string => typeof q === 'string' && q.trim() !== '').slice(0, 3)
      : []
  }
}
