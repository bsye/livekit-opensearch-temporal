import {
  condition,
  patched,
  setHandler,
  sleep,
  startChild,
  upsertSearchAttributes,
  workflowInfo,
  type ChildWorkflowHandle,
  type SignalDefinition,
} from '@temporalio/workflow';
import {
  agentMetrics,
  livekitEvent,
  ParticipantIdentities,
  participantState,
  RoomName,
  roomState,
  transcript,
  type LiveKitEvent,
  type ParticipantSessionState,
  type RoomState,
} from './shared.js';

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
 */
export async function roomSession(): Promise<RoomState & { endReason: string }> {
  // Rooms started before participant lanes existed replay without them
  const withLanes = patched('participant-lanes');
  const state: RoomState = {
    participants: {},
    egress: {},
    transcript: [],
    metrics: [],
    eventCount: 0,
    duplicateCount: 0,
  };
  const seen = new Set<string>();
  const lanes = new ParticipantLanes(() => state.name);
  let finished = false;

  setHandler(livekitEvent, (e) => {
    // LiveKit may deliver a webhook more than once
    if (seen.has(e.id)) {
      state.duplicateCount++;
      return;
    }
    seen.add(e.id);
    state.eventCount++;
    apply(state, e);
    if (withLanes && e.participant) lanes.forward(e.participant.identity, e.participant.kind, livekitEvent, e);
    if (e.event === 'room_finished') finished = true;
  });
  setHandler(transcript, (entry) => {
    state.transcript.push(entry);
    if (withLanes) lanes.forward(entry.participant, undefined, transcript, entry);
  });
  setHandler(agentMetrics, (metric) => {
    state.metrics.push(metric);
    if (withLanes && metric.participant) lanes.forward(metric.participant, 'AGENT', agentMetrics, metric);
  });
  setHandler(roomState, () => state);

  const roomFinished = await condition(() => finished, MAX_SESSION);
  if (roomFinished) await sleep(LATE_EVENT_GRACE);
  return { ...state, endReason: roomFinished ? 'room_finished' : 'timeout' };
}

/** One participant in one room session; completes shortly after they leave. */
export async function participantSession(init: {
  roomSid: string;
  identity: string;
  kind?: string;
}): Promise<ParticipantSessionState & { endReason: string }> {
  const state: ParticipantSessionState = { ...init, sid: '', tracks: {}, transcript: [], metrics: [] };
  let left = false;

  setHandler(livekitEvent, (e) => {
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
  });
  setHandler(transcript, (entry) => void state.transcript.push(entry));
  setHandler(agentMetrics, (metric) => void state.metrics.push(metric));
  setHandler(participantState, () => state);

  const hasLeft = await condition(() => left, MAX_SESSION);
  if (hasLeft) await sleep(PARTICIPANT_LATE_EVENT_GRACE);
  return { ...state, endReason: hasLeft ? 'left' : 'timeout' };
}

/**
 * Routes signals to one participantSession child per participant identity, starting it on
 * first use. Each participant's signals are chained so they arrive in order.
 */
class ParticipantLanes {
  private lanes = new Map<string, Promise<ChildWorkflowHandle<typeof participantSession> | undefined>>();

  constructor(private roomName: () => string | undefined) {}

  forward<T>(identity: string, kind: string | undefined, signal: SignalDefinition<[T]>, arg: T): void {
    const previous = this.lanes.get(identity) ?? this.start(identity, kind);
    const next = previous.then(async (child) => {
      // A lane that already completed (late event after its grace period) just drops the signal
      await child?.signal(signal, arg).catch(() => undefined);
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

/** Timeline label prefix per LiveKit participant kind (protobuf JSON enum names). */
function actorIcon(kind: string | undefined): string {
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
