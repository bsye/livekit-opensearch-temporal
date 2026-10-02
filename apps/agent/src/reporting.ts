import { metrics, voice } from '@livekit/agents';
import type { Client } from '@temporalio/client';
import { signalRoom, type ActionDecision, type MemoryRoute, type RoomRef, type ToolStep, type Turn } from '@voice/temporal';

/** One pipeline measurement, folded into the current turn's timing. */
export type AgentMetric =
  | { type: 'stt'; speechMs: number; transcriptionMs: number }
  | { type: 'eou'; endOfTurnMs: number }
  | { type: 'llm'; firstTokenMs: number }
  | { type: 'tts'; firstAudioMs: number }
  | { type: 'voice_to_voice'; ms: number };

/**
 * What the agent tells Temporal. A turn is collected as it happens (the user's words, possibly over
 * several fragments; the memory route; tool calls, approvals and Laya audits; where the time went)
 * and sent as one signal when the agent's reply is complete: the room shows it as one
 * conversationTurn row. A failed signal never breaks the call.
 */
export class RoomReporter {
  private turn = emptyTurn();

  constructor(
    private readonly temporal: Client,
    private readonly room: () => RoomRef,
    private readonly agent: () => string,
  ) {}

  userSaid(text: string, participant: string, at: number): void {
    if (!this.turn.userText) this.turn.startedAt = at;
    this.turn.user = participant;
    this.turn.userText = [this.turn.userText, text].filter(Boolean).join(' ');
  }

  memory(route: MemoryRoute): void {
    this.turn.memory = route;
  }

  tool(step: ToolStep): void {
    this.turn.tools.push(step);
  }

  /** Approval steps and Laya audits (audits can land just after the reply: they join the next turn then). */
  action = (decision: ActionDecision): void => {
    this.turn.actions.push(decision);
  };

  /** Tool durations by tool call id, filled in by the tools as they run. */
  readonly toolTimings = new Map<string, number>();

  metric(m: AgentMetric): void {
    const t = this.turn.timing;
    switch (m.type) {
      case 'stt': // one per speech segment: a turn can have several
        t.speechMs = (t.speechMs ?? 0) + m.speechMs;
        t.transcriptionMs = m.transcriptionMs;
        break;
      case 'eou':
        t.endOfTurnMs = m.endOfTurnMs;
        break;
      case 'llm': // the first call; later ones follow tool results
        if (m.firstTokenMs >= 0) t.firstTokenMs ??= m.firstTokenMs;
        break;
      case 'tts':
        t.firstAudioMs ??= m.firstAudioMs;
        break;
      case 'voice_to_voice':
        t.voiceToVoiceMs ??= m.ms;
        break;
    }
  }

  /** The agent finished (or was cut off in) its reply: the turn is complete. */
  agentSaid(text: string, at: number, opts: { interrupted?: boolean; fromMemory?: boolean } = {}): void {
    const turn: Turn = {
      ...this.turn,
      agent: this.agent(),
      reply: text,
      endedAt: at,
      startedAt: this.turn.startedAt || at,
      interrupted: opts.interrupted || undefined,
      fromMemory: opts.fromMemory || undefined,
    };
    this.turn = emptyTurn();
    signalRoom(this.temporal, this.room(), { type: 'turn', data: turn }).catch((err) =>
      console.error('temporal turn signal failed', err),
    );
  }
}

function emptyTurn(): Omit<Turn, 'agent' | 'reply' | 'endedAt'> {
  return { user: '', userText: '', startedAt: 0, tools: [], actions: [], timing: {} };
}

/**
 * Wires an AgentSession (cascade, omni) to the reporter: transcripts become turns, and executed tools
 * and pipeline timings join the current turn.
 */
export function reportSession(
  session: voice.AgentSession,
  reporter: RoomReporter,
  opts: {
    user: () => string;
    /** VAD reports the end of speech this late; added back to voice-to-voice latency. */
    vadSilenceMs: number;
    /** Text as reported (e.g. without markers the agent adds to user messages). */
    clean?: (text: string) => string;
    /** Whether the reply being completed was answered from memory (read, then reset). */
    takeFromMemory?: () => boolean;
    onUserText?: (text: string) => void;
  },
): void {
  const clean = opts.clean ?? ((t: string) => t);
  session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item, createdAt }) => {
    if (item.type !== 'message' || !item.textContent) return;
    const text = clean(item.textContent);
    if (item.role === 'user') {
      opts.onUserText?.(text);
      console.log(`user: ${text}`);
      reporter.userSaid(text, opts.user(), createdAt);
    } else if (item.role === 'assistant') {
      console.log(`assistant: ${text}`);
      reporter.agentSaid(text, createdAt, { interrupted: item.interrupted, fromMemory: opts.takeFromMemory?.() });
    }
  });

  session.on(voice.AgentSessionEventTypes.FunctionToolsExecuted, (event) => {
    for (const [call, output] of voice.zipFunctionCallsAndOutputs(event)) {
      reporter.tool({
        name: call.name,
        args: safeJson(call.args),
        output: output?.output,
        isError: output?.isError || undefined,
        at: event.createdAt ?? Date.now(),
        durationMs: reporter.toolTimings.get(call.callId),
      });
    }
  });

  // the TTS StreamAdapter re-emits the wrapped TTS's metrics, so each request arrives twice
  const reportedRequests = new Set<string>();
  session.on(voice.AgentSessionEventTypes.MetricsCollected, ({ metrics: m }) => {
    if ('requestId' in m && m.requestId) {
      const key = `${m.type}:${m.requestId}`;
      if (reportedRequests.has(key)) return;
      reportedRequests.add(key);
    }
    metrics.logMetrics(m);
    const metric = toAgentMetric(m);
    if (metric) reporter.metric(metric);
  });

  // voice-to-voice: user stops speaking → agent starts
  let userStoppedAt: number | undefined;
  session.on(voice.AgentSessionEventTypes.UserStateChanged, ({ oldState, newState, createdAt }) => {
    if (oldState === 'speaking' && newState === 'listening') userStoppedAt = createdAt;
  });
  session.on(voice.AgentSessionEventTypes.AgentStateChanged, ({ newState, createdAt }) => {
    if (newState !== 'speaking' || userStoppedAt === undefined) return;
    const durationMs = createdAt - userStoppedAt + opts.vadSilenceMs;
    userStoppedAt = undefined;
    console.log(`voice-to-voice latency: ${durationMs}ms`);
    reporter.metric({ type: 'voice_to_voice', ms: durationMs });
  });
}

function toAgentMetric(m: metrics.AgentMetrics): AgentMetric | undefined {
  switch (m.type) {
    case 'stt_metrics':
      return { type: 'stt', speechMs: m.audioDurationMs, transcriptionMs: m.durationMs };
    case 'eou_metrics':
      return { type: 'eou', endOfTurnMs: m.endOfUtteranceDelayMs };
    case 'llm_metrics':
      return { type: 'llm', firstTokenMs: m.ttftMs };
    case 'tts_metrics':
      return { type: 'tts', firstAudioMs: m.ttfbMs };
    default:
      return undefined; // VAD metrics fire continuously
  }
}

function safeJson(s: string): Record<string, unknown> {
  try {
    return JSON.parse(s) as Record<string, unknown>;
  } catch {
    return { raw: s };
  }
}
