export interface LiveKitEvent {
  id: string
  event: string
  createdAt?: string
  room?: { sid: string; name: string }
  participant?: { sid: string; identity: string; name?: string; kind?: string }
  track?: { sid: string; type?: string; source?: string; name?: string; mimeType?: string }
  egressInfo?: { egressId: string; roomId?: string; roomName?: string; status?: string }
  ingressInfo?: { ingressId: string; roomName?: string }
  [key: string]: unknown
}

export interface TrackState {
  sid: string
  type?: string
  source?: string
  mimeType?: string
  publishedAt?: string
  unpublishedAt?: string
}

export interface ParticipantState {
  sid: string
  identity: string
  kind?: string
  joinedAt?: string
  leftAt?: string
  tracks: Record<string, TrackState>
}

export interface ActionDecision {
  tool: string
  args: Record<string, unknown>
  decision: 'confirm' | 'confirmed' | 'declined' | 'audited' | 'flagged'
  reasons: string[]
  intent?: number
  latencyMs: number
  at: number
  participant: string
}

export interface ReminderInput {
  text: string
  fireAt: number
  when: string
  roomSid: string
  participant: string
}

export interface EmailInput {
  to: string
  subject: string
  body: string
  roomSid: string
  requestedBy: string
}

export interface ToolStep {
  name: string
  args: Record<string, unknown>
  output?: string
  isError?: boolean
  at: number
  durationMs?: number
}

export interface TurnTiming {
  speechMs?: number
  transcriptionMs?: number
  endOfTurnMs?: number
  firstTokenMs?: number
  firstAudioMs?: number
  voiceToVoiceMs?: number
}

export interface MemoryRoute {
  route: string
  text?: string
  ms?: number
}

export interface Turn {
  index?: number
  user: string
  userText: string
  startedAt: number
  agent: string
  reply: string
  endedAt: number
  interrupted?: boolean
  fromMemory?: boolean
  memory?: MemoryRoute
  tools: ToolStep[]
  actions: ActionDecision[]
  timing: TurnTiming
}

export interface RoomState {
  sid?: string
  name?: string
  startedAt?: string
  finishedAt?: string
  participants: Record<string, ParticipantState>
  egress: Record<string, { status?: string }>
  turns: Turn[]
  eventCount: number
  duplicateCount: number
}

export type RoomSignal = { type: 'livekitEvent'; data: LiveKitEvent } | { type: 'turn'; data: Turn }

export interface RoomRef {
  sid: string
  name?: string
}
