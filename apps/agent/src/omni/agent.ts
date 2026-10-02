import { fileURLToPath } from 'node:url'
import {
  cli,
  defineAgent,
  inference,
  type JobContext,
  type JobProcess,
  type llm,
  ServerOptions,
  voice,
} from '@livekit/agents'
import * as openai from '@livekit/agents-plugin-openai'
import * as silero from '@livekit/agents-plugin-silero'
import type { AudioFrame } from '@livekit/rtc-node'
import { env } from '@voice/config'
import { warmReranker } from '@voice/memory'
import { connectTemporal, type RoomRef } from '@voice/temporal'
import { RoomReporter, reportSession } from '../reporting.js'
import { createActions } from '../tools/actions.js'
import { ActionAuditor } from '../tools/audit.js'
import { createTools } from '../tools/index.js'
import { RecallPrefetch } from '../tools/prefetch.js'
import { exampleChatCtx, instructions } from './prompt.js'
import { TurnAudio } from './turn-audio.js'

const LOCAL_API_KEY = 'local'

const clean = (text: string) =>
  text
    .replace(/\s*<audio:[0-9a-f]+>/g, '')
    .replace(/\s*<memory>[\s\S]*?<\/memory>/g, '')
    .trim()

process.env.LLM_BASE_URL = env('OMNI_BASE_URL')
process.env.LLM_MODEL = env('OMNI_MODEL')

const VAD_SILENCE_MS = 300
const ENDPOINTING_MIN_DELAY_MS = 300
const ENDPOINTING_MAX_DELAY_MS = 2500
const INTERRUPTION_MIN_MS = 600
const INTERRUPTION_MIN_WORDS = 2

class OmniAgent extends voice.Agent {
  constructor(
    opts: ConstructorParameters<typeof voice.Agent>[0],
    private readonly turnAudio: TurnAudio,
  ) {
    super(opts)
  }

  async sttNode(audio: ReadableStream<AudioFrame> | AsyncIterable<AudioFrame>, settings: voice.ModelSettings) {
    return voice.Agent.default.sttNode(this, this.turnAudio.tap(audio), settings)
  }

  async onUserTurnCompleted(_chatCtx: llm.ChatContext, newMessage: llm.ChatMessage) {
    const id = await this.turnAudio.upload().catch((err) => {
      console.error('turn audio upload failed; the model reads the transcript instead', err)
      return undefined
    })
    if (id) newMessage.content.push(` <audio:${id}>`)
  }
}

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load({ minSilenceDuration: VAD_SILENCE_MS })
    await warmReranker()
  },

  entry: async (ctx: JobContext) => {
    const temporal = await connectTemporal()
    const turnAudio = new TurnAudio(env('OMNI_BASE_URL'))

    const session = new voice.AgentSession({
      vad: ctx.proc.userData.vad as silero.VAD,
      stt: new openai.STT({
        baseURL: env('SPEECH_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('STT_MODEL'),
        language: 'en',
        useRealtime: false,
      }),
      llm: new openai.LLM({
        baseURL: env('OMNI_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('OMNI_MODEL'),
        temperature: 0,
      }),
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
        preemptiveGeneration: { enabled: false },
      },
    })

    let room: RoomRef = { sid: '', name: '' }
    let agentIdentity = 'agent'
    const reporter = new RoomReporter(
      temporal,
      () => room,
      () => agentIdentity,
    )

    const userTurns: string[] = []
    let lastUser = 'user'
    const currentUser = () => (lastUser = ctx.room.remoteParticipants.values().next().value?.identity ?? lastUser)
    let answeringFromMemory = false

    const auditor = new ActionAuditor(
      () => agentIdentity,
      () => userTurns.slice(-2).join(' '),
      reporter.action,
    )
    const actions = createActions({ temporal, room: () => room, user: currentUser, auditor })
    const prefetch = new RecallPrefetch(() => room.sid)
    const tools = createTools(actions, {
      room: () => room,
      agent: () => agentIdentity,
      reportAction: reporter.action,
      onRecall: () => (answeringFromMemory = true),
      toolTimings: reporter.toolTimings,
      prefetch,
    })

    const agent = new OmniAgent({ instructions: instructions(new Date()), tools, chatCtx: exampleChatCtx() }, turnAudio)
    prefetch.attach(session)
    await session.start({ agent, room: ctx.room })
    await ctx.connect()

    room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' }
    agentIdentity = ctx.room.localParticipant?.identity ?? 'agent'

    reportSession(session, reporter, {
      user: currentUser,
      vadSilenceMs: VAD_SILENCE_MS,
      clean,
      onUserText: (text) => userTurns.push(text),
      takeFromMemory: () => {
        const was = answeringFromMemory
        answeringFromMemory = false
        return was
      },
    })
    session.on(voice.AgentSessionEventTypes.UserStateChanged, ({ newState, createdAt }) => {
      if (newState === 'speaking') turnAudio.speechStarted(createdAt)
    })

    session.say('Hi! How can I help you today?')
  },
})

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }))
}
