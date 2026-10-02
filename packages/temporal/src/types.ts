/** A LiveKit webhook (WebhookEvent.toJson()); only the fields the workflows read are typed. */
export interface LiveKitEvent {
  id: string;
  event: string;
  createdAt?: string;
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

export interface TranscriptEntry {
  role: 'user' | 'assistant';
  text: string;
  participant: string;
  at: number;
  interrupted?: boolean;
  /** The assistant answered from a memory recall. */
  fromMemory?: boolean;
}

export interface AgentMetric {
  type: 'stt_metrics' | 'eou_metrics' | 'llm_metrics' | 'tts_metrics' | 'turn_latency';
  at: number;
  participant?: string;
  durationMs?: number;
  ttftMs?: number;
  ttfbMs?: number;
  endOfUtteranceDelayMs?: number;
  transcriptionDelayMs?: number;
  audioDurationMs?: number;
  promptTokens?: number;
  completionTokens?: number;
}

/**
 * A step in the life of a tool call: the user is asked (confirm) and answers (confirmed/declined),
 * then Laya audits what ran (audited/flagged).
 */
export interface ActionDecision {
  tool: string;
  args: Record<string, unknown>;
  decision: 'confirm' | 'confirmed' | 'declined' | 'audited' | 'flagged';
  reasons: string[];
  /** Laya's P(the action matches the request). */
  intent?: number;
  latencyMs: number;
  at: number;
  participant: string;
}

export interface ReminderInput {
  text: string;
  fireAt: number;
  when: string;
  roomSid: string;
  participant: string;
}

export interface EmailInput {
  to: string;
  subject: string;
  body: string;
  roomSid: string;
  requestedBy: string;
}

export interface RoomState {
  sid?: string;
  name?: string;
  startedAt?: string;
  finishedAt?: string;
  participants: Record<string, ParticipantState>;
  egress: Record<string, { status?: string }>;
  transcript: TranscriptEntry[];
  metrics: AgentMetric[];
  actions: ActionDecision[];
  eventCount: number;
  duplicateCount: number;
}

export interface ParticipantSessionState extends ParticipantState {
  roomSid: string;
  transcript: TranscriptEntry[];
  metrics: AgentMetric[];
  actions: ActionDecision[];
}

export type RoomSignal =
  | { type: 'livekitEvent'; data: LiveKitEvent }
  | { type: 'transcript'; data: TranscriptEntry }
  | { type: 'agentMetrics'; data: AgentMetric }
  | { type: 'action'; data: ActionDecision };

export interface RoomRef {
  sid: string;
  name?: string;
}
