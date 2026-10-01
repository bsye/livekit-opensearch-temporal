// Shared between the translator (client), the worker and the workflow sandbox:
// only import from @temporalio/common and @temporalio/workflow here.
import { defineSearchAttributeKey, SearchAttributeType } from '@temporalio/common';
import { defineQuery, defineSignal } from '@temporalio/workflow';

export const TASK_QUEUE = 'livekit-rooms';

// Registered by infra/temporal/setup.sh
export const RoomName = defineSearchAttributeKey('RoomName', SearchAttributeType.KEYWORD);
export const ParticipantIdentities = defineSearchAttributeKey(
  'ParticipantIdentities',
  SearchAttributeType.KEYWORD_LIST,
);

/**
 * A LiveKit webhook event in protobuf JSON form (WebhookEvent.toJson()).
 * Only the fields the workflow reads are typed; the full payload is kept in history.
 */
export interface LiveKitEvent {
  id: string;
  event: string; // room_started, participant_joined, track_published, ...
  createdAt?: string; // unix seconds (int64 as string)
  room?: { sid: string; name: string };
  participant?: { sid: string; identity: string; name?: string; kind?: string };
  track?: { sid: string; type?: string; source?: string; name?: string; mimeType?: string };
  egressInfo?: { egressId: string; roomId?: string; roomName?: string; status?: string };
  ingressInfo?: { ingressId: string; roomName?: string };
  [key: string]: unknown;
}

export interface TrackState {
  sid: string;
  type?: string;
  source?: string;
  mimeType?: string;
  publishedAt?: string;
  unpublishedAt?: string;
}

export interface ParticipantState {
  sid: string;
  identity: string;
  kind?: string;
  joinedAt?: string;
  leftAt?: string;
  tracks: Record<string, TrackState>;
}

/** A finalized conversation turn, sent by the agent (agents/voice-agent). */
export interface TranscriptEntry {
  role: string; // user | assistant
  text: string;
  participant: string; // identity of the speaker (the agent's own identity for assistant turns)
  at: number; // unix ms
  interrupted?: boolean;
}

/** One pipeline stage measurement from the agent (STT, end-of-utterance, LLM, TTS). */
export interface AgentMetric {
  type: string; // stt_metrics | eou_metrics | llm_metrics | tts_metrics | turn_latency (voice-to-voice)
  at: number; // unix ms
  participant?: string; // identity of the agent that measured it
  durationMs?: number;
  ttftMs?: number; // LLM time to first token
  ttfbMs?: number; // TTS time to first byte
  endOfUtteranceDelayMs?: number;
  transcriptionDelayMs?: number;
  audioDurationMs?: number;
  promptTokens?: number;
  completionTokens?: number;
}

export interface RoomState {
  sid?: string;
  name?: string;
  startedAt?: string;
  finishedAt?: string;
  participants: Record<string, ParticipantState>; // by identity
  egress: Record<string, { status?: string }>;
  transcript: TranscriptEntry[];
  metrics: AgentMetric[];
  eventCount: number;
  duplicateCount: number;
}

/** One participant's lane: a participantSession child workflow per participant per room. */
export interface ParticipantSessionState extends ParticipantState {
  roomSid: string;
  transcript: TranscriptEntry[]; // what this participant said
  metrics: AgentMetric[]; // pipeline metrics, for agents
}

// Fixed-name signals: what senders used before labelled signals; still handled for those rooms
export const livekitEvent = defineSignal<[LiveKitEvent]>('livekitEvent');
export const transcript = defineSignal<[TranscriptEntry]>('transcript');
export const agentMetrics = defineSignal<[AgentMetric]>('agentMetrics');
export const roomState = defineQuery<RoomState>('roomState');
export const participantState = defineQuery<ParticipantSessionState>('participantState');

/**
 * Every signal to a room or participant workflow carries one of these envelopes. The signal
 * *name* is a human-readable label (see signalLabel) because the Temporal UI timeline labels
 * signals by name only; workflows accept any name through a default signal handler.
 */
export type RoomSignal =
  | { type: 'livekitEvent'; data: LiveKitEvent }
  | { type: 'transcript'; data: TranscriptEntry }
  | { type: 'agentMetrics'; data: AgentMetric };

/** Icon per LiveKit participant kind (protobuf JSON enum names). */
export function actorIcon(kind: string | undefined): string {
  switch (kind) {
    case 'AGENT':
      return '🤖';
    case 'INGRESS':
      return '📥';
    case 'EGRESS':
      return '📤';
    case 'SIP':
      return '☎️';
    default:
      return '👤';
  }
}

/** Timeline label: who did what, e.g. `👤 dalbi · track_published (AUDIO)` or `🤖 agent-… · llm first token 530ms`. */
export function signalLabel(s: RoomSignal): string {
  switch (s.type) {
    case 'livekitEvent': {
      const e = s.data;
      if (e.participant) {
        // protobuf JSON omits default enum values, and AUDIO is the default track type
        const track = e.track ? ` (${e.track.type ?? 'AUDIO'})` : '';
        return `${actorIcon(e.participant.kind)} ${e.participant.identity} · ${e.event}${track}`;
      }
      if (e.egressInfo) return `📤 egress · ${e.event}`;
      if (e.ingressInfo) return `📥 ingress · ${e.event}`;
      return `🏠 ${e.event}`;
    }
    case 'transcript': {
      const t = s.data;
      const text = t.text.length > 60 ? `${t.text.slice(0, 57)}…` : t.text;
      return `${t.role === 'assistant' ? '🤖' : '👤'} ${t.participant}: “${text}”${t.interrupted ? ' (interrupted)' : ''}`;
    }
    case 'agentMetrics': {
      const m = s.data;
      const ms = (v: number | undefined) => `${Math.round(v ?? 0)}ms`;
      const what =
        m.type === 'stt_metrics' ? `stt ${ms(m.durationMs)}`
        : m.type === 'eou_metrics' ? `end of turn ${ms(m.endOfUtteranceDelayMs)}`
        : m.type === 'llm_metrics' ? `llm first token ${ms(m.ttftMs)}`
        : m.type === 'tts_metrics' ? `tts first audio ${ms(m.ttfbMs)}`
        : m.type === 'turn_latency' ? `⏱ voice-to-voice ${ms(m.durationMs)}`
        : m.type;
      return `🤖 ${m.participant ?? 'agent'} · ${what}`;
    }
  }
}
