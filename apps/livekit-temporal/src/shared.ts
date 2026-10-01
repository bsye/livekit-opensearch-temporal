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

export const livekitEvent = defineSignal<[LiveKitEvent]>('livekitEvent');
export const transcript = defineSignal<[TranscriptEntry]>('transcript');
export const agentMetrics = defineSignal<[AgentMetric]>('agentMetrics');
export const roomState = defineQuery<RoomState>('roomState');
export const participantState = defineQuery<ParticipantSessionState>('participantState');
