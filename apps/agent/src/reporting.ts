import { metrics, voice } from '@livekit/agents'
import type { Client } from '@temporalio/client'
import {
  type ActionDecision,
  type MemoryRoute,
  type RoomRef,
  signalRoom,
  type ToolStep,
  type Turn,
} from '@voice/temporal'

export type AgentMetric =
  | { type: 'stt'; speechMs: number; transcriptionMs: number }
  | { type: 'eou'; endOfTurnMs: number }
  | { type: 'llm'; firstTokenMs: number }
  | { type: 'tts'; firstAudioMs: number }
  | { type: 'voice_to_voice'; ms: number }

export class RoomReporter {
  private turn = emptyTurn()

  constructor(
    private readonly temporal: Client,
    private readonly room: () => RoomRef,
    private readonly agent: () => string,
  ) {}

  userSaid(text: string, participant: string, at: number): void {
    if (!this.turn.userText) this.turn.startedAt = at
    this.turn.user = participant
    this.turn.userText = [this.turn.userText, text].filter(Boolean).join(' ')
  }

  memory(route: MemoryRoute): void {
    this.turn.memory = route
  }

  tool(step: ToolStep): void {
    this.turn.tools.push(step)
  }

  action = (decision: ActionDecision): void => {
    this.turn.actions.push(decision)
  }

  readonly toolTimings = new Map<string, number>()

  metric(m: AgentMetric): void {
    const t = this.turn.timing
    switch (m.type) {
      case 'stt':
        t.speechMs = (t.speechMs ?? 0) + m.speechMs
        t.transcriptionMs = m.transcriptionMs
        break
      case 'eou':
        t.endOfTurnMs = m.endOfTurnMs
        break
      case 'llm':
        if (m.firstTokenMs >= 0) t.firstTokenMs ??= m.firstTokenMs
        break
      case 'tts':
        t.firstAudioMs ??= m.firstAudioMs
        break
      case 'voice_to_voice':
        t.voiceToVoiceMs ??= m.ms
        break
    }
  }

  agentSaid(text: string, at: number, opts: { interrupted?: boolean; fromMemory?: boolean } = {}): void {
    const turn: Turn = {
      ...this.turn,
      agent: this.agent(),
      reply: text,
      endedAt: at,
      startedAt: this.turn.startedAt || at,
      interrupted: opts.interrupted || undefined,
      fromMemory: opts.fromMemory || undefined,
    }
    this.turn = emptyTurn()
    signalRoom(this.temporal, this.room(), { type: 'turn', data: turn }).catch((err) =>
      console.error('temporal turn signal failed', err),
    )
  }
}

function emptyTurn(): Omit<Turn, 'agent' | 'reply' | 'endedAt'> {
  return { user: '', userText: '', startedAt: 0, tools: [], actions: [], timing: {} }
}

export function reportSession(
  session: voice.AgentSession,
  reporter: RoomReporter,
  opts: {
    user: () => string
    vadSilenceMs: number
    clean?: (text: string) => string
    takeFromMemory?: () => boolean
    onUserText?: (text: string) => void
  },
): void {
  const clean = opts.clean ?? ((t: string) => t)
  session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item, createdAt }) => {
    if (item.type !== 'message' || !item.textContent) return
    const text = clean(item.textContent)
    if (item.role === 'user') {
      opts.onUserText?.(text)
      console.log(`user: ${text}`)
      reporter.userSaid(text, opts.user(), createdAt)
    } else if (item.role === 'assistant') {
      console.log(`assistant: ${text}`)
      reporter.agentSaid(text, createdAt, { interrupted: item.interrupted, fromMemory: opts.takeFromMemory?.() })
    }
  })

  session.on(voice.AgentSessionEventTypes.FunctionToolsExecuted, (event) => {
    for (const [call, output] of voice.zipFunctionCallsAndOutputs(event)) {
      reporter.tool({
        name: call.name,
        args: safeJson(call.args),
        output: output?.output,
        isError: output?.isError || undefined,
        at: event.createdAt ?? Date.now(),
        durationMs: reporter.toolTimings.get(call.callId),
      })
    }
  })

  const reportedRequests = new Set<string>()
  session.on(voice.AgentSessionEventTypes.MetricsCollected, ({ metrics: m }) => {
    if ('requestId' in m && m.requestId) {
      const key = `${m.type}:${m.requestId}`
      if (reportedRequests.has(key)) return
      reportedRequests.add(key)
    }
    metrics.logMetrics(m)
    const metric = toAgentMetric(m)
    if (metric) reporter.metric(metric)
  })

  let userStoppedAt: number | undefined
  session.on(voice.AgentSessionEventTypes.UserStateChanged, ({ oldState, newState, createdAt }) => {
    if (oldState === 'speaking' && newState === 'listening') userStoppedAt = createdAt
  })
  session.on(voice.AgentSessionEventTypes.AgentStateChanged, ({ newState, createdAt }) => {
    if (newState !== 'speaking' || userStoppedAt === undefined) return
    const durationMs = createdAt - userStoppedAt + opts.vadSilenceMs
    userStoppedAt = undefined
    console.log(`voice-to-voice latency: ${durationMs}ms`)
    reporter.metric({ type: 'voice_to_voice', ms: durationMs })
  })
}

function toAgentMetric(m: metrics.AgentMetrics): AgentMetric | undefined {
  switch (m.type) {
    case 'stt_metrics':
      return { type: 'stt', speechMs: m.audioDurationMs, transcriptionMs: m.durationMs }
    case 'eou_metrics':
      return { type: 'eou', endOfTurnMs: m.endOfUtteranceDelayMs }
    case 'llm_metrics':
      return { type: 'llm', firstTokenMs: m.ttftMs }
    case 'tts_metrics':
      return { type: 'tts', firstAudioMs: m.ttfbMs }
    default:
      return undefined
  }
}

function safeJson(s: string): Record<string, unknown> {
  try {
    return JSON.parse(s) as Record<string, unknown>
  } catch {
    return { raw: s }
  }
}
