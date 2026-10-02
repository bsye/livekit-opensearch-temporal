import { fileURLToPath } from 'node:url'
import { cli, defineAgent, type JobContext, ServerOptions } from '@livekit/agents'
import {
  AudioFrame,
  AudioSource,
  AudioStream,
  LocalAudioTrack,
  type RemoteTrack,
  RoomEvent,
  TrackKind,
  TrackPublishOptions,
  TrackSource,
} from '@livekit/rtc-node'
import { env } from '@voice/config'
import { route } from '@voice/laya'
import { recallBrief, warmReranker } from '@voice/memory'
import { connectTemporal } from '@voice/temporal'
import { isOpen, removeTags, trimStartChars } from '@voice/text'
import { FRAME_SAMPLES, parseToolCalls, stripToolResponses, type ToolCall, VoiceChatSession } from '@voice/voicechat'
import { RoomReporter } from '../reporting.js'
import { createActions, type EmailArgs, type ReminderArgs } from '../tools/actions.js'
import { ActionAuditor } from '../tools/audit.js'
import { prompt } from './prompt.js'
import { isFirstPerson, refersBack } from './routing.js'

const INPUT_RATE = 16_000
const OUTPUT_RATE = 22_050
const ROUTE_AFTER_QUIET_MS = 700
const TURN_QUIET_MS = 1200
const MAX_HOLD_MS = 2500
const MAX_CALLS_PER_TURN = 3
const STRONG_MATCH = 4

export default defineAgent({
  prewarm: async () => {
    await warmReranker()
  },

  entry: async (ctx: JobContext) => {
    const temporal = await connectTemporal()
    await ctx.connect()
    const room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' }
    const agentIdentity = ctx.room.localParticipant?.identity ?? 'agent'
    let user = 'user'
    const reporter = new RoomReporter(
      temporal,
      () => room,
      () => agentIdentity,
    )
    const reportAction = reporter.action

    const started = Date.now()
    const model = await VoiceChatSession.open(env('VOICECHAT_URL'), prompt())
    console.log(`voicechat session ready in ${Date.now() - started}ms`)

    let userText = ''
    let answeringFromMemory = false
    const auditor = new ActionAuditor(
      () => agentIdentity,
      () => userText,
      reportAction,
    )
    const actions = createActions({ temporal, room: () => room, user: () => user, auditor })

    async function runTool(call: ToolCall): Promise<string> {
      const args = call.arguments as Record<string, string>
      switch (call.name) {
        case 'set_reminder':
          return actions.setReminder(args as unknown as ReminderArgs)
        case 'cancel_reminder':
          return actions.cancelReminder(args)
        case 'recall':
          answeringFromMemory = true
          return (await recallBrief({ question: args.question ?? userText, excludeRoomSid: room.sid })).text
        case 'send_email': {
          const confirmed = call.arguments.confirmed === true
          reportAction({
            tool: 'send_email',
            args: call.arguments,
            decision: confirmed ? 'confirmed' : 'confirm',
            reasons: [],
            latencyMs: 0,
            at: Date.now(),
            participant: agentIdentity,
          })
          if (!confirmed) return 'Not sent yet. Read the email back to the user and ask them to confirm.'
          return actions.sendEmail(args as unknown as EmailArgs)
        }
        default:
          return `There is no tool called ${call.name}.`
      }
    }

    let handledCalls = 0
    let turnCalls = 0
    let turnRecalled = false
    let toolQueue: Promise<void> = Promise.resolve()
    const modelCalls = () => parseToolCalls(stripToolResponses(model.functionText))
    const callOpen = () => isOpen(model.functionText, '<TOOLCALL>', '</TOOLCALL>')

    model.onFunction = () => {
      const calls = modelCalls()
      for (; handledCalls < calls.length; handledCalls++) {
        const call = calls[handledCalls]
        console.log(`tool call ${call.name} ${JSON.stringify(call.arguments)}`)
        turnCalls++
        const repeatRecall = call.name === 'recall' && turnRecalled
        if (call.name === 'recall') turnRecalled = true
        toolQueue = toolQueue.then(async () => {
          const t = Date.now()
          const out =
            turnCalls > MAX_CALLS_PER_TURN
              ? 'Too many tool calls. Stop calling tools and answer the user now.'
              : repeatRecall
                ? 'Already searched for this turn: answer from the result above.'
                : call.name === '(unparsable)'
                  ? 'That tool call was not valid JSON. Answer the user without it.'
                  : await runTool(call).catch((err) => `The tool failed: ${err}`)
          await model.toolOutput(out)
          reporter.tool({ name: call.name, args: call.arguments, output: out, at: t, durationMs: Date.now() - t })
          console.log(`tool ${call.name} → "${out.slice(0, 160)}" (${Date.now() - t}ms incl. injecting)`)
        })
      }
    }

    const source = new AudioSource(OUTPUT_RATE, 1)
    await ctx.room.localParticipant!.publishTrack(
      LocalAudioTrack.createAudioTrack('voicechat', source),
      new TrackPublishOptions({ source: TrackSource.SOURCE_MICROPHONE }),
    )
    const play = (pcm: Int16Array) =>
      void source.captureFrame(new AudioFrame(pcm, OUTPUT_RATE, 1, pcm.length)).catch(() => undefined)
    let held: Int16Array[] | null = null
    let holdTimer: NodeJS.Timeout | undefined
    const holdAudio = () => {
      if (held) return
      held = []
      holdTimer = setTimeout(() => releaseAudio(true), MAX_HOLD_MS)
    }
    const releaseAudio = (playHeld: boolean) => {
      clearTimeout(holdTimer)
      const frames = held ?? []
      held = null
      if (playHeld) frames.forEach(play)
    }
    model.onAudio = (pcm) => (held ? held.push(pcm) : play(pcm))

    let userFrom = 0
    let agentFrom = 0
    let routedTurn = -1
    let routeTimer: NodeJS.Timeout | undefined

    async function routeTurn() {
      const heard = trimStartChars(userText, ' \n,.?!;:-')
      if (!heard || routedTurn === userFrom) return
      routedTurn = userFrom
      holdAudio()
      const callsBefore = modelCalls().length
      const t = Date.now()
      const [laya, found] = await Promise.all([
        route(heard).catch(() => 'chat' as const),
        recallBrief({ question: heard, excludeRoomSid: room.sid }).catch((err) => {
          console.error('recall failed', err)
          return { text: '', top: -Infinity, hits: 0 }
        }),
      ])
      if (!found.text) return releaseAudio(true)
      const firstPerson = isFirstPerson(heard)
      const backReference = refersBack(heard)
      const memoryTurn = laya === 'past' || backReference || (firstPerson && found.top >= STRONG_MATCH)
      console.log(
        `route "${heard}": laya ${laya}, best match ${found.top.toFixed(1)} → ${memoryTurn ? 'memory' : 'no memory'} (${Date.now() - t}ms)`,
      )
      reporter.memory({ route: laya, text: memoryTurn ? found.text : undefined, ms: Date.now() - t })
      const worthForcing = memoryTurn && (found.hits > 0 || firstPerson || backReference)
      if (!worthForcing || modelCalls().length > callsBefore || callOpen()) return releaseAudio(true)
      handledCalls++
      turnCalls++
      turnRecalled = true
      answeringFromMemory = true
      toolQueue = toolQueue.then(async () => {
        await model.toolOutput(found.text, { name: 'recall', arguments: { question: heard } })
        releaseAudio(false)
        agentFrom = model.assistantText.length
        console.log(`forced recall → "${found.text.slice(0, 160)}" (${Date.now() - t}ms)`)
      })
    }

    let lastUserAt = 0
    let lastAgentAt = 0
    let latencyPending = false
    model.onUserText = (text) => {
      if (!userText) [turnCalls, turnRecalled] = [0, false]
      lastUserAt = Date.now()
      userText = text.slice(userFrom).trim()
      latencyPending = true
      clearTimeout(routeTimer)
      routeTimer = setTimeout(() => void routeTurn(), ROUTE_AFTER_QUIET_MS)
    }

    const reply = () => model.assistantText.slice(agentFrom).trim()
    const turnTimer = setInterval(() => {
      const now = Date.now()
      const lastText = model.textEvents.at(-1)
      if (latencyPending && lastText && lastText.at > lastUserAt) {
        latencyPending = false
        void routeTurn()
        const durationMs = lastText.at - lastUserAt
        console.log(`voice-to-voice latency: ${durationMs}ms`)
        reporter.metric({ type: 'voice_to_voice', ms: durationMs })
      }
      if (lastText) lastAgentAt = lastText.at
      if (
        userText &&
        now - lastUserAt > TURN_QUIET_MS &&
        (lastAgentAt > lastUserAt || now - lastUserAt > 3 * TURN_QUIET_MS)
      ) {
        reporter.userSaid(userText, user, lastUserAt)
        userFrom = model.userText.length
        userText = ''
      }
      if (reply() && now - lastAgentAt > TURN_QUIET_MS && now - lastUserAt > TURN_QUIET_MS) {
        const text = removeTags(reply()).trim()
        if (text) reporter.agentSaid(text, lastAgentAt, { fromMemory: answeringFromMemory })
        agentFrom = model.assistantText.length
        answeringFromMemory = false
      }
    }, 200)

    let listening = false
    const listen = async (track: RemoteTrack, identity: string) => {
      if (listening || track.kind !== TrackKind.KIND_AUDIO) return
      listening = true
      user = identity
      console.log(`listening to ${identity}`)
      let buffer = new Int16Array(0)
      for await (const frame of new AudioStream(track, { sampleRate: INPUT_RATE, numChannels: 1 })) {
        const merged = new Int16Array(buffer.length + frame.data.length)
        merged.set(buffer)
        merged.set(frame.data, buffer.length)
        let at = 0
        for (; at + FRAME_SAMPLES <= merged.length; at += FRAME_SAMPLES)
          model.push(merged.slice(at, at + FRAME_SAMPLES))
        buffer = merged.slice(at)
      }
      listening = false
    }
    ctx.room.on(RoomEvent.TrackSubscribed, (track, _pub, participant) => void listen(track, participant.identity))
    for (const p of ctx.room.remoteParticipants.values()) {
      for (const pub of p.trackPublications.values()) if (pub.track) void listen(pub.track as RemoteTrack, p.identity)
    }

    ctx.addShutdownCallback(async () => {
      clearInterval(turnTimer)
      model.close()
    })
  },
})

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }))
}
