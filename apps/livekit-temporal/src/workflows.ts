import {
  condition,
  patched,
  proxyActivities,
  setDefaultSignalHandler,
  setHandler,
  sleep,
  startChild,
  upsertSearchAttributes,
  workflowInfo,
  type ChildWorkflowHandle,
} from '@temporalio/workflow';
import {
  actorIcon,
  agentMetrics,
  livekitEvent,
  ParticipantIdentities,
  participantState,
  RoomName,
  roomState,
  transcript,
  type LiveKitEvent,
  type ParticipantSessionState,
  type EmailInput,
  type ReminderInput,
  type RoomSignal,
  type RoomState,
  type TranscriptEntry,
} from './shared.js';
import type * as activities from './activities.js';

// Webhooks can arrive after room_finished; keep the workflow open briefly to record them.
const LATE_EVENT_GRACE = '1 minute';
// Same for a participant lane: the agent's last transcript/metrics arrive after the user leaves.
const PARTICIPANT_LATE_EVENT_GRACE = '10 seconds';
// Safety net if room_finished is never delivered (webhooks are not guaranteed).
const MAX_SESSION = '24 hours';

/**
 * One workflow per LiveKit room session (workflow id = room sid). Keeps the combined room
 * state and starts a participantSession child per participant, so each actor gets its own
 * labelled lane in the Temporal UI timeline.
 *
 * Signals arrive under human-readable names (signalLabel) with a RoomSignal payload, and are
 * accepted through the default signal handler; the fixed-name handlers serve older senders.
 */
export async function roomSession(): Promise<RoomState & { endReason: string }> {
  // Rooms started before these features existed replay without them
  const withLanes = patched('participant-lanes');
  const labelled = patched('labelled-signals');
  const state: RoomState = {
    participants: {},
    egress: {},
    transcript: [],
    metrics: [],
    eventCount: 0,
    duplicateCount: 0,
  };
  const seen = new Set<string>();
  const lanes = new ParticipantLanes(() => state.name, labelled);
  const memory = new MemoryIndexer(() => state.sid ?? workflowInfo().workflowId, () => state.name);
  const withMemory = patched('memory-index');
  let finished = false;

  // label = the signal name, forwarded unchanged so the participant lane shows the same text
  const onSignal = (signal: RoomSignal, label: string) => {
    switch (signal.type) {
      case 'livekitEvent': {
        const e = signal.data;
        // LiveKit may deliver a webhook more than once
        if (seen.has(e.id)) {
          state.duplicateCount++;
          return;
        }
        seen.add(e.id);
        state.eventCount++;
        apply(state, e);
        if (withLanes && e.participant) lanes.forward(e.participant.identity, e.participant.kind, signal, label);
        if (e.event === 'room_finished') finished = true;
        break;
      }
      case 'transcript':
        state.transcript.push(signal.data);
        if (withLanes) lanes.forward(signal.data.participant, undefined, signal, label);
        if (withMemory) memory.add(signal.data);
        break;
      case 'agentMetrics':
        state.metrics.push(signal.data);
        if (withLanes && signal.data.participant) lanes.forward(signal.data.participant, 'AGENT', signal, label);
        break;
      case 'gate':
        (state.gates ??= []).push(signal.data);
        if (withLanes) lanes.forward(signal.data.participant, 'AGENT', signal, label);
        break;
    }
  };
  setDefaultSignalHandler((name, payload) => {
    if (isRoomSignal(payload)) onSignal(payload, name);
  });
  setHandler(livekitEvent, (data) => onSignal({ type: 'livekitEvent', data }, 'livekitEvent'));
  setHandler(transcript, (data) => onSignal({ type: 'transcript', data }, 'transcript'));
  setHandler(agentMetrics, (data) => onSignal({ type: 'agentMetrics', data }, 'agentMetrics'));
  setHandler(roomState, () => state);

  const roomFinished = labelled
    ? await condition(() => finished, MAX_SESSION, { summary: '⏳ waiting for room_finished (24h max)' })
    : await condition(() => finished, MAX_SESSION);
  if (roomFinished) {
    await sleep(LATE_EVENT_GRACE, labelled ? { summary: '⏳ grace period for late webhooks' } : undefined);
  }
  return { ...state, endReason: roomFinished ? 'room_finished' : 'timeout' };
}

/** One participant in one room session; completes shortly after they leave. */
export async function participantSession(init: {
  roomSid: string;
  identity: string;
  kind?: string;
}): Promise<ParticipantSessionState & { endReason: string }> {
  const labelled = patched('labelled-signals');
  const state: ParticipantSessionState = { ...init, sid: '', tracks: {}, transcript: [], metrics: [] };
  let left = false;

  const onSignal = (signal: RoomSignal) => {
    switch (signal.type) {
      case 'livekitEvent': {
        const e = signal.data;
        if (e.participant) {
          state.sid = e.participant.sid;
          state.kind = e.participant.kind ?? state.kind;
        }
        switch (e.event) {
          case 'participant_joined':
            state.joinedAt = e.createdAt;
            break;
          case 'participant_left':
          case 'participant_connection_aborted':
            state.leftAt = e.createdAt;
            left = true;
            break;
          case 'track_published':
          case 'track_unpublished': {
            if (!e.track) break;
            const t = (state.tracks[e.track.sid] ??= { sid: e.track.sid });
            Object.assign(t, { type: e.track.type, source: e.track.source, mimeType: e.track.mimeType });
            if (e.event === 'track_published') t.publishedAt = e.createdAt;
            else t.unpublishedAt = e.createdAt;
            break;
          }
        }
        break;
      }
      case 'transcript':
        state.transcript.push(signal.data);
        break;
      case 'agentMetrics':
        state.metrics.push(signal.data);
        break;
      case 'gate':
        (state.gates ??= []).push(signal.data);
        break;
    }
  };
  setDefaultSignalHandler((_name, payload) => {
    if (isRoomSignal(payload)) onSignal(payload);
  });
  setHandler(livekitEvent, (data) => onSignal({ type: 'livekitEvent', data }));
  setHandler(transcript, (data) => onSignal({ type: 'transcript', data }));
  setHandler(agentMetrics, (data) => onSignal({ type: 'agentMetrics', data }));
  setHandler(participantState, () => state);

  const hasLeft = labelled
    ? await condition(() => left, MAX_SESSION, { summary: '⏳ in the room (24h max)' })
    : await condition(() => left, MAX_SESSION);
  if (hasLeft) {
    await sleep(PARTICIPANT_LATE_EVENT_GRACE, labelled ? { summary: '⏳ grace period for late events' } : undefined);
  }
  return { ...state, endReason: hasLeft ? 'left' : 'timeout' };
}

/** A reminder set by the agent's set_reminder tool: a durable timer that fires at fireAt. */
export async function reminder(input: ReminderInput): Promise<ReminderInput & { firedAt: number }> {
  const wait = input.fireAt - Date.now(); // Date.now() is replay-safe inside workflows
  if (wait > 0) await sleep(wait, { summary: `⏰ until ${input.when}` });
  // Delivery (push, call back into the room, …) is a next step; firing is recorded in history
  return { ...input, firedAt: Date.now() };
}

/**
 * Groups the transcript into exchanges (user turn(s) + the agent's reply) and indexes each one
 * into conversation memory as an activity: a labelled row on the room's timeline.
 */
class MemoryIndexer {
  private userTurns: TranscriptEntry[] = [];

  constructor(
    private roomSid: () => string,
    private roomName: () => string | undefined,
  ) {}

  add(entry: TranscriptEntry): void {
    if (entry.role !== 'assistant') {
      this.userTurns.push(entry);
      return;
    }
    if (this.userTurns.length === 0) return; // greeting or follow-up with no user turn to pair
    const turns = this.userTurns;
    this.userTurns = [];
    const userText = turns.map((t) => t.text).join(' ');
    const { indexConversationExchange } = proxyActivities<typeof activities>({
      startToCloseTimeout: '30 seconds',
      retry: { maximumAttempts: 5 },
      summary: `🧠 remember: “${userText.length > 50 ? `${userText.slice(0, 47)}…` : userText}”`,
    });
    void indexConversationExchange({
      roomSid: this.roomSid(),
      roomName: this.roomName(),
      user: turns[0].participant,
      agent: entry.participant,
      userText,
      agentText: entry.text,
      startedAt: turns[0].at,
      endedAt: entry.at,
    }).catch(() => undefined); // indexing failures must not fail the room session
  }
}

/** Simulated outbox: records an approved email; a real version would call an email activity. */
export async function sendEmail(input: EmailInput): Promise<EmailInput & { sentAt: number; simulated: true }> {
  return { ...input, sentAt: Date.now(), simulated: true };
}

/**
 * Routes signals to one participantSession child per participant identity, starting it on
 * first use. Each participant's signals are chained so they arrive in order.
 */
class ParticipantLanes {
  private lanes = new Map<string, Promise<ChildWorkflowHandle<typeof participantSession> | undefined>>();

  constructor(
    private roomName: () => string | undefined,
    private labelled: boolean,
  ) {}

  forward(identity: string, kind: string | undefined, signal: RoomSignal, label: string): void {
    const previous = this.lanes.get(identity) ?? this.start(identity, kind);
    const next = previous.then(async (child) => {
      // Rooms started before labelled signals forward under the fixed names
      const sent = this.labelled ? child?.signal(label, signal) : child?.signal(signal.type, signal.data);
      // A lane that already completed (late event after its grace period) just drops the signal
      await sent?.catch(() => undefined);
      return child;
    });
    this.lanes.set(identity, next);
  }

  private start(identity: string, kind: string | undefined) {
    const roomSid = workflowInfo().workflowId;
    const name = this.roomName();
    return startChild(participantSession, {
      workflowId: `${roomSid}/${identity}`,
      args: [{ roomSid, identity, kind }],
      staticSummary: `${actorIcon(kind)} ${identity}`,
      typedSearchAttributes: name ? [{ key: RoomName, value: name }] : [],
    }).catch(() => undefined); // e.g. a lane for this identity already ran in this room: skip it
  }
}

function isRoomSignal(payload: unknown): payload is RoomSignal {
  const type = (payload as RoomSignal | undefined)?.type;
  return type === 'livekitEvent' || type === 'transcript' || type === 'agentMetrics' || type === 'gate';
}

function apply(state: RoomState, e: LiveKitEvent): void {
  if (e.room) {
    state.sid = e.room.sid;
    state.name = e.room.name;
  }
  switch (e.event) {
    case 'room_started':
      state.startedAt = e.createdAt;
      break;
    case 'room_finished':
      state.finishedAt = e.createdAt;
      break;
    case 'participant_joined': {
      const p = participant(state, e);
      if (!p) break;
      p.joinedAt = e.createdAt;
      upsertSearchAttributes([{ key: ParticipantIdentities, value: Object.keys(state.participants) }]);
      break;
    }
    case 'participant_left':
    case 'participant_connection_aborted': {
      const p = participant(state, e);
      if (p) p.leftAt = e.createdAt;
      break;
    }
    case 'track_published':
    case 'track_unpublished': {
      const p = participant(state, e);
      if (!p || !e.track) break;
      const t = (p.tracks[e.track.sid] ??= { sid: e.track.sid });
      Object.assign(t, { type: e.track.type, source: e.track.source, mimeType: e.track.mimeType });
      if (e.event === 'track_published') t.publishedAt = e.createdAt;
      else t.unpublishedAt = e.createdAt;
      break;
    }
    default:
      if (e.egressInfo) state.egress[e.egressInfo.egressId] = { status: e.egressInfo.status };
  }
}

function participant(state: RoomState, e: LiveKitEvent) {
  if (!e.participant) return undefined;
  const { identity, sid, kind } = e.participant;
  return (state.participants[identity] ??= { identity, sid, kind, tracks: {} });
}
