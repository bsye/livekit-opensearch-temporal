import { fileURLToPath } from 'node:url'
import { cli, defineAgent, type JobContext, ServerOptions, voice } from '@livekit/agents'
import { env } from '@voice/config'
import { localLlm, prewarm, startLocalAgent } from '../pipeline.js'
import { stripControlTokens } from './llm-filter.js'

const instructions = (now: Date) => `You are a helpful voice assistant running fully on local models.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.
Use tools only when the user asks for that action. After a tool runs, tell the user what it did.
Whenever the user asks about their own life, past, plans, purchases, people they know, or anything they
told you before, call recall first, every time, even if you searched earlier in this conversation.
Never say you have no information about the user's past without calling recall.
If the user corrects a reminder or says undo, cancel it (and set the corrected one).
The current local time is ${now.toTimeString().slice(0, 5)}, ${now.toDateString()}.`

export default defineAgent({
  prewarm,

  entry: async (ctx: JobContext) => {
    const session = await startLocalAgent(ctx, {
      llm: localLlm(env('LLM_BASE_URL'), env('LLM_MODEL'), { reasoningEffort: 'none' }),
      preemptiveGeneration: true,
      createAgent: (tools) =>
        voice.Agent.create({
          instructions: instructions(new Date()),
          tools,
          llmNode: async (agentCtx, chatCtx, toolCtx, settings) => {
            const stream = await voice.Agent.default.llmNode(agentCtx.agent, chatCtx, toolCtx, settings)
            return stream ? stripControlTokens(stream) : null
          },
        }),
    })
    session.generateReply({ instructions: 'Greet the user in one short sentence.' })
  },
})

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }))
}
