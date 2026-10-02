import { inference, type JobContext, type JobProcess, type llm, voice } from '@livekit/agents'
import * as openai from '@livekit/agents-plugin-openai'
import * as silero from '@livekit/agents-plugin-silero'
import { env } from '@voice/config'
import { warmReranker } from '@voice/memory'
import { connectTemporal, type RoomRef } from '@voice/temporal'
import { RoomReporter, reportSession } from './reporting.js'
import { createActions } from './tools/actions.js'
import { ActionAuditor } from './tools/audit.js'
import { createTools } from './tools/index.js'
import { RecallPrefetch } from './tools/prefetch.js'

const LOCAL_API_KEY = 'local'
const VAD_SILENCE_MS = 300
const ENDPOINTING_MIN_DELAY_MS = 300
const ENDPOINTING_MAX_DELAY_MS = 2500
const INTERRUPTION_MIN_MS = 600
const INTERRUPTION_MIN_WORDS = 2

export async function prewarm(proc: JobProcess) {
  proc.userData.vad = await silero.VAD.load({ minSilenceDuration: VAD_SILENCE_MS })
  await warmReranker()
}

export function localLlm(baseUrl: string, model: string, options: { reasoningEffort?: 'none'; temperature?: number }) {
  return new openai.LLM({ baseURL: baseUrl, apiKey: LOCAL_API_KEY, model, ...options })
}

export type Tools = ReturnType<typeof createTools>

export interface LocalAgentOptions {
  llm: llm.LLM
  preemptiveGeneration: boolean
  createAgent: (tools: Tools) => voice.Agent
  cleanUserText?: (text: string) => string
}

export async function startLocalAgent(ctx: JobContext, options: LocalAgentOptions): Promise<voice.AgentSession> {
  const temporal = await connectTemporal()
  const session = new voice.AgentSession({
    vad: ctx.proc.userData.vad as silero.VAD,
    stt: new openai.STT({
      baseURL: env('SPEECH_BASE_URL'),
      apiKey: LOCAL_API_KEY,
      model: env('STT_MODEL'),
      language: 'en',
      useRealtime: false,
    }),
    llm: options.llm,
    tts: new openai.TTS({
      baseURL: env('SPEECH_BASE_URL'),
      apiKey: LOCAL_API_KEY,
      model: env('TTS_MODEL'),
      voice: env('TTS_VOICE') as openai.TTSVoices,
    }),
    turnHandling: {
      turnDetection: new inference.TurnDetector({ version: 'v1-mini' }),
      endpointing: { minDelay: ENDPOINTING_MIN_DELAY_MS, maxDelay: ENDPOINTING_MAX_DELAY_MS },
      interruption: { mode: 'vad', minDuration: INTERRUPTION_MIN_MS, minWords: INTERRUPTION_MIN_WORDS },
      preemptiveGeneration: options.preemptiveGeneration ? { enabled: true, preemptiveTts: false } : { enabled: false },
    },
  })

  let room: RoomRef = { sid: '', name: '' }
  let agentIdentity = 'agent'
  let lastUser = 'user'
  let answeringFromMemory = false
  const userTurns: string[] = []
  const currentUser = () => {
    lastUser = ctx.room.remoteParticipants.values().next().value?.identity ?? lastUser
    return lastUser
  }

  const reporter = new RoomReporter(
    temporal,
    () => room,
    () => agentIdentity,
  )
  const auditor = new ActionAuditor(
    () => agentIdentity,
    () => userTurns.slice(-2).join(' '),
    reporter.action,
  )
  const prefetch = new RecallPrefetch(() => room.sid)
  const tools = createTools(createActions({ temporal, room: () => room, user: currentUser, auditor }), {
    room: () => room,
    agent: () => agentIdentity,
    reportAction: reporter.action,
    onRecall: () => {
      answeringFromMemory = true
    },
    toolTimings: reporter.toolTimings,
    prefetch,
  })

  prefetch.attach(session)
  await session.start({ agent: options.createAgent(tools), room: ctx.room })
  await ctx.connect()
  room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' }
  agentIdentity = ctx.room.localParticipant?.identity ?? 'agent'

  reportSession(session, reporter, {
    user: currentUser,
    vadSilenceMs: VAD_SILENCE_MS,
    clean: options.cleanUserText,
    onUserText: (text) => userTurns.push(text),
    takeFromMemory: () => {
      const was = answeringFromMemory
      answeringFromMemory = false
      return was
    },
  })
  return session
}
