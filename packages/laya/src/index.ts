import { env } from '@voice/config'

export type Question =
  | { type: 'noul'; instructions: string }
  | { type: 'choice'; instructions: string; criteria: Record<string, string> }

export interface Answer {
  noul?: number
  choice?: string
  confidence?: number
}

export async function ask<K extends string>(
  state: string,
  questions: Record<K, Question>,
): Promise<{ answers: Record<K, Answer>; latencyMs: number }> {
  const body = (await post('/v1/systemone', { state, questions })) as { answers: Record<K, Answer>; latency_ms: number }
  return { answers: body.answers, latencyMs: body.latency_ms }
}

export async function scan(states: string[], question: string): Promise<{ scores: number[]; peakMb?: number }> {
  const body = (await post('/v1/scan', { states, question })) as { scores: number[]; peak_mb?: number }
  return { scores: body.scores, peakMb: body.peak_mb }
}

export type Route = 'past' | 'action' | 'chat'

export const ROUTE_QUESTION: Question = {
  type: 'choice',
  instructions: 'What does the user want from the assistant in this turn?',
  criteria: {
    A: 'Information from their own past or from earlier conversations with the assistant',
    B: 'An action: a reminder, timer, message, email, calendar entry, call or other task',
    C: 'Conversation, general knowledge, advice or anything else',
  },
}

const ROUTES: Record<string, Route> = { A: 'past', B: 'action', C: 'chat' }

export async function route(text: string): Promise<Route> {
  const { answers } = await ask(`User: ${text}`, { q: ROUTE_QUESTION })
  return ROUTES[answers.q.choice ?? ''] ?? 'chat'
}

async function post(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${env('LAYA_BASE_URL')}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`laya ${path}: ${res.status} ${await res.text()}`)
  return res.json()
}
