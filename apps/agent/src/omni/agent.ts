import { fileURLToPath } from 'node:url'
import { cli, defineAgent, type JobContext, type llm, ServerOptions, voice } from '@livekit/agents'
import type { AudioFrame } from '@livekit/rtc-node'
import { env } from '@voice/config'
import { removeBetween } from '@voice/text'
import { localLlm, prewarm, startLocalAgent } from '../pipeline.js'
import { exampleChatCtx, instructions } from './prompt.js'
import { TurnAudio } from './turn-audio.js'

const withoutMarkers = (text: string) =>
  removeBetween(removeBetween(text, '<audio:', '>'), '<memory>', '</memory>').trim()

process.env.LLM_BASE_URL = env('OMNI_BASE_URL')
process.env.LLM_MODEL = env('OMNI_MODEL')

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
  prewarm,

  entry: async (ctx: JobContext) => {
    const turnAudio = new TurnAudio(env('OMNI_BASE_URL'))
    const session = await startLocalAgent(ctx, {
      llm: localLlm(env('OMNI_BASE_URL'), env('OMNI_MODEL'), { temperature: 0 }),
      preemptiveGeneration: false,
      cleanUserText: withoutMarkers,
      createAgent: (tools) =>
        new OmniAgent({ instructions: instructions(new Date()), tools, chatCtx: exampleChatCtx() }, turnAudio),
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
