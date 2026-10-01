// Query expansion: the LLM lists the words the user most likely used when they first talked about
// what the question asks (synonyms, specific nouns), never the answer. BM25 then searches the
// question plus these terms. Cached per question so reruns don't call the LLM again.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const LLM = process.env.LLM_BASE_URL ?? 'http://localhost:1234/v1';
const MODEL = process.env.LLM_MODEL ?? 'google/gemma-4-26b-a4b';
const SYSTEM =
  'You generate search keywords. Given a question a user asks about their own past chats, output the words and short ' +
  'phrases they most likely used when they first talked about it: specific nouns, synonyms, related terms. Never guess ' +
  'the answer. Output only 10 to 20 comma-separated terms.';
const EXAMPLE_Q = 'Question: What was the name of the restaurant I liked in Rome?';
const EXAMPLE_A = 'restaurant, Rome, Italy, dinner, trattoria, pizzeria, pasta, food, ate, meal, trip, vacation, recommend, favorite, loved';

export class QueryExpander {
  private cache: Record<string, string>;

  constructor(private cacheFile: URL) {
    this.cache = existsSync(cacheFile) ? (JSON.parse(readFileSync(cacheFile, 'utf8')) as Record<string, string>) : {};
  }

  async expand(questionId: string, question: string): Promise<string> {
    const hit = this.cache[questionId];
    if (hit !== undefined) return hit;
    const res = await fetch(`${LLM}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        reasoning_effort: 'none',
        temperature: 0,
        max_tokens: 120,
        messages: [
          { role: 'system', content: SYSTEM },
          { role: 'user', content: EXAMPLE_Q },
          { role: 'assistant', content: EXAMPLE_A },
          { role: 'user', content: `Question: ${question}` },
        ],
      }),
    });
    if (!res.ok) throw new Error(`expansion: ${res.status} ${await res.text()}`);
    const text = ((await res.json()) as { choices: { message: { content: string } }[] }).choices[0].message.content.trim();
    this.cache[questionId] = text;
    writeFileSync(this.cacheFile, JSON.stringify(this.cache));
    return text;
  }
}
