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

/** One tool the agent ran in a turn, with what it returned. */
export interface ToolStep {
  name: string;
  args: Record<string, unknown>;
  output?: string;
  isError?: boolean;
  at: number;
  durationMs?: number;
}

/** Where a turn's time went: the user's side (speaking, being recognised) and the agent's. */
export interface TurnTiming {
  /** How long the user spoke. */
  speechMs?: number;
  /** Speech-to-text compute for the last segment. */
  transcriptionMs?: number;
  /** End of speech → turn confirmed (silence, turn detector, transcription). */
  endOfTurnMs?: number;
  /** The first model call's time to first token. */
  firstTokenMs?: number;
  /** Text-to-speech time to first audio. */
  firstAudioMs?: number;
  /** User stops speaking → agent starts speaking. */
  voiceToVoiceMs?: number;
}

/** How the agent decided whether a turn needed memory (Laya route + recall). */
export interface MemoryRoute {
  route: string;
  /** What the model was given (absent when the turn didn't use memory). */
  text?: string;
  ms?: number;
}

/**
 * One exchange: what the user said (possibly over several fragments), how the agent handled it,
 * and its reply. Sent by the agent when the reply is complete; the room starts a conversationTurn
 * child for each, so the room's timeline reads as one row per turn.
 */
export interface Turn {
  /** Assigned by the room: 1, 2, 3, ... (0: the agent spoke first, e.g. its greeting). */
  index?: number;
  user: string;
  userText: string;
  startedAt: number;
  agent: string;
  reply: string;
  endedAt: number;
  interrupted?: boolean;
  /** The reply was answered from memory: not new evidence for later recalls. */
  fromMemory?: boolean;
  memory?: MemoryRoute;
  tools: ToolStep[];
  actions: ActionDecision[];
  timing: TurnTiming;
}

export interface RoomState {
  sid?: string;
  name?: string;
  startedAt?: string;
  finishedAt?: string;
  participants: Record<string, ParticipantState>;
  egress: Record<string, { status?: string }>;
  turns: Turn[];
  eventCount: number;
  duplicateCount: number;
}

export type RoomSignal = { type: 'livekitEvent'; data: LiveKitEvent } | { type: 'turn'; data: Turn };

export interface RoomRef {
  sid: string;
  name?: string;
}
