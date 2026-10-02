import { fileURLToPath } from 'node:url'
import { cli, defineAgent, inference, type JobContext, type JobProcess, ServerOptions, voice } from '@livekit/agents'
import * as openai from '@livekit/agents-plugin-openai'
import * as silero from '@livekit/agents-plugin-silero'
import { env } from '@voice/config'
import { warmReranker } from '@voice/memory'
import { connectTemporal, type RoomRef } from '@voice/temporal'
import { RoomReporter, reportSession } from '../reporting.js'
import { createActions } from '../tools/actions.js'
import { ActionAuditor } from '../tools/audit.js'
import { createTools } from '../tools/index.js'
import { RecallPrefetch } from '../tools/prefetch.js'
import { stripControlTokens } from './llm-filter.js'

const instructions = (now: Date) => `You are a helpful voice assistant running fully on local models.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.
Use tools only when the user asks for that action. After a tool runs, tell the user what it did.
Whenever the user asks about their own life, past, plans, purchases, people they know, or anything they
told you before, call recall first, every time, even if you searched earlier in this conversation.
Never say you have no information about the user's past without calling recall.
If the user corrects a reminder or says undo, cancel it (and set the corrected one).
The current local time is ${now.toTimeString().slice(0, 5)}, ${now.toDateString()}.`

const LOCAL_API_KEY = 'local' // the OpenAI client requires one; local servers ignore it

// Turn-taking. VAD reports end of speech after VAD_SILENCE_MS; the audio turn detector then decides
// from intonation whether the user is done or pausing. Silence alone either split sentences at
// natural pauses or, tuned longer, slowed every reply.
const VAD_SILENCE_MS = 300
const ENDPOINTING_MIN_DELAY_MS = 300
const ENDPOINTING_MAX_DELAY_MS = 2500
// Only real speech interrupts the agent, not "okay" or a blip of echo.
const INTERRUPTION_MIN_MS = 600
const INTERRUPTION_MIN_WORDS = 2

/** STT → LLM → TTS, all on local models. */
export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load({ minSilenceDuration: VAD_SILENCE_MS })
    await warmReranker()
  },

  entry: async (ctx: JobContext) => {
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
      llm: new openai.LLM({
        baseURL: env('LLM_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('LLM_MODEL'),
        reasoningEffort: 'none', // thinking adds seconds per turn
      }),
      tts: new openai.TTS({
        baseURL: env('SPEECH_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('TTS_MODEL'),
        voice: env('TTS_VOICE') as openai.TTSVoices,
      }),
      turnHandling: {
        // dev mode defaults to v1, which only runs on LiveKit Cloud
        turnDetection: new inference.TurnDetector({ version: 'v1-mini' }),
        endpointing: { minDelay: ENDPOINTING_MIN_DELAY_MS, maxDelay: ENDPOINTING_MAX_DELAY_MS },
        interruption: { mode: 'vad', minDuration: INTERRUPTION_MIN_MS, minWords: INTERRUPTION_MIN_WORDS },
        // the LLM starts on the final transcript, but TTS waits for the confirmed turn, or discarded drafts get spoken
        preemptiveGeneration: { enabled: true, preemptiveTts: false },
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
    // transcripts can arrive after the user left, when the room no longer lists them
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

    prefetch.attach(session)
    await session.start({
      agent: voice.Agent.create({
        instructions: instructions(new Date()),
        tools,
        llmNode: async (agentCtx, chatCtx, toolCtx, settings) => {
          const stream = await voice.Agent.default.llmNode(agentCtx.agent, chatCtx, toolCtx, settings)
          return stream ? stripControlTokens(stream) : null
        },
      }),
      room: ctx.room,
    })
    await ctx.connect()

    room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' }
    agentIdentity = ctx.room.localParticipant?.identity ?? 'agent'

    reportSession(session, reporter, {
      user: currentUser,
      vadSilenceMs: VAD_SILENCE_MS,
      onUserText: (text) => userTurns.push(text),
      takeFromMemory: () => {
        const was = answeringFromMemory
        answeringFromMemory = false
        return was
      },
    })

    session.generateReply({ instructions: 'Greet the user in one short sentence.' })
  },
})

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }))
}
