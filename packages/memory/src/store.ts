import { env } from '@voice/config'
import { requestJson } from '@voice/http'
import { ask } from '@voice/laya'

export const TOPICS = {
  work: 'work, projects, meetings, colleagues',
  personal: 'family, friends, home, plans',
  travel: 'trips, flights, hotels, places',
  health: 'health, doctors, fitness',
  money: 'money, payments, bills, shopping',
  tech: 'software, computers, technical topics',
  smalltalk: 'greetings, chit-chat, thanks',
} as const
export type Topic = keyof typeof TOPICS

export interface Exchange {
  roomSid: string
  roomName?: string
  user: string
  agent: string
  userText: string
  agentText: string
  startedAt: number
  endedAt: number
  fromMemory?: boolean
}

export interface MemoryDoc extends Exchange {
  text: string
  topic: Topic
  topicConfidence: number
  hasTask: boolean
}

export interface SearchHit {
  _id: string
  _score: number
  _source: MemoryDoc
}

const MAPPING = {
  mappings: {
    properties: {
      roomSid: { type: 'keyword' },
      roomName: { type: 'keyword' },
      user: { type: 'keyword' },
      agent: { type: 'keyword' },
      userText: { type: 'text', analyzer: 'english' },
      agentText: { type: 'text', analyzer: 'english' },
      text: { type: 'text', analyzer: 'english' },
      startedAt: { type: 'date', format: 'epoch_millis' },
      endedAt: { type: 'date', format: 'epoch_millis' },
      topic: { type: 'keyword' },
      topicConfidence: { type: 'float' },
      hasTask: { type: 'boolean' },
      fromMemory: { type: 'boolean' },
    },
  },
}

let indexReady: Promise<void> | undefined

export function ensureIndex(): Promise<void> {
  indexReady ??= (async () => {
    if ((await fetch(indexUrl(), { method: 'HEAD' })).ok) return
    const res = await fetch(indexUrl(), {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(MAPPING),
    })
    if (!res.ok && !(await res.text()).includes('resource_already_exists_exception')) {
      throw new Error(`create index ${env('MEMORY_INDEX')}: ${res.status}`)
    }
  })().catch((err) => {
    indexReady = undefined
    throw err
  })
  return indexReady
}

export function exchangeText(e: Pick<Exchange, 'userText' | 'agentText'>): string {
  return `User: ${e.userText}\nAssistant: ${e.agentText}`
}

export async function categorise(text: string): Promise<Pick<MemoryDoc, 'topic' | 'topicConfidence' | 'hasTask'>> {
  const { answers } = await ask(text, {
    topic: { type: 'choice', instructions: 'What is this conversation mainly about?', criteria: TOPICS },
    task: { type: 'noul', instructions: 'Does the user ask for a task, reminder or commitment?' },
  })
  return {
    topic: answers.topic.choice as Topic,
    topicConfidence: answers.topic.confidence ?? 0,
    hasTask: (answers.task.noul ?? 0) >= 0.5,
  }
}

export async function toMemoryDoc(e: Exchange): Promise<MemoryDoc> {
  const text = exchangeText(e)
  return { ...e, text, ...(await categorise(text)) }
}

export const docId = (e: Pick<Exchange, 'roomSid' | 'endedAt'>) => `${e.roomSid}-${e.endedAt}`

export async function indexExchange(e: Exchange): Promise<MemoryDoc> {
  await ensureIndex()
  const doc = await toMemoryDoc(e)
  await request('PUT', `${indexUrl()}/_doc/${encodeURIComponent(docId(e))}`, doc)
  return doc
}

export async function bulkIndex(docs: MemoryDoc[]): Promise<void> {
  if (!docs.length) return
  const index = env('MEMORY_INDEX')
  const lines = docs.flatMap((doc) => [
    JSON.stringify({ index: { _index: index, _id: docId(doc) } }),
    JSON.stringify(doc),
  ])
  const res = (await request(
    'POST',
    `${env('OPENSEARCH_URL')}/_bulk`,
    `${lines.join('\n')}\n`,
    'application/x-ndjson',
  )) as {
    errors: boolean
  }
  if (res.errors) throw new Error('bulk indexing errors')
}

export async function deleteRoom(roomName: string): Promise<void> {
  await request('POST', `${indexUrl()}/_delete_by_query?refresh=true`, { query: { term: { roomName } } })
}

export async function refresh(): Promise<void> {
  await request('POST', `${indexUrl()}/_refresh`, {})
}

export async function search(body: unknown): Promise<SearchHit[]> {
  const res = (await request('POST', `${indexUrl()}/_search`, body)) as { hits: { hits: SearchHit[] } }
  return res.hits.hits
}

const indexUrl = () => `${env('OPENSEARCH_URL')}/${env('MEMORY_INDEX')}`

const request = requestJson<unknown>
