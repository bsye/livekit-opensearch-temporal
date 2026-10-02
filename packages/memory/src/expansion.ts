import { env } from '@voice/config';

const SYSTEM =
  'You generate search keywords. Given a question a user asks about their own past chats, output the words and short ' +
  'phrases they most likely used when they first talked about it: specific nouns, synonyms, related terms. Never guess ' +
  'the answer. Output only 10 to 20 comma-separated terms.';
const EXAMPLE_QUESTION = 'Question: What was the name of the restaurant I liked in Rome?';
const EXAMPLE_TERMS =
  'restaurant, Rome, Italy, dinner, trattoria, pizzeria, pasta, food, ate, meal, trip, vacation, recommend, favorite, loved';

/** The words the user most likely used back then, so BM25 can match a paraphrased question. */
export async function expandQuery(
  question: string,
  llm: { baseUrl: string; model: string } = { baseUrl: env('LLM_BASE_URL'), model: env('LLM_MODEL') },
): Promise<string> {
  const res = await fetch(`${llm.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: llm.model,
      reasoning_effort: 'none',
      temperature: 0,
      max_tokens: 120,
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: EXAMPLE_QUESTION },
        { role: 'assistant', content: EXAMPLE_TERMS },
        { role: 'user', content: `Question: ${question}` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`query expansion: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { choices: { message: { content: string } }[] }).choices[0].message.content.trim();
}
