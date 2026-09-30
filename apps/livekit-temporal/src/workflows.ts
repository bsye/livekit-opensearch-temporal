import { condition, setHandler, sleep, upsertSearchAttributes } from '@temporalio/workflow';
import {
  agentMetrics,
  livekitEvent,
  ParticipantIdentities,
  roomState,
  transcript,
  type LiveKitEvent,
  type RoomState,
} from './shared.js';

// Webhooks can arrive after room_finished; keep the workflow open briefly to record them.
const LATE_EVENT_GRACE = '1 minute';
// Safety net if room_finished is never delivered (webhooks are not guaranteed).
const MAX_SESSION = '24 hours';

/** One workflow per LiveKit room session (workflow id = room sid). */
export async function roomSession(): Promise<RoomState & { endReason: string }> {
  const state: RoomState = {
    participants: {},
    egress: {},
    transcript: [],
    metrics: [],
    eventCount: 0,
    duplicateCount: 0,
  };
  const seen = new Set<string>();
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
    if (e.event === 'room_finished') finished = true;
  });
  setHandler(transcript, (entry) => void state.transcript.push(entry));
  setHandler(agentMetrics, (metric) => void state.metrics.push(metric));
  setHandler(roomState, () => state);

  const roomFinished = await condition(() => finished, MAX_SESSION);
  if (roomFinished) await sleep(LATE_EVENT_GRACE);
  return { ...state, endReason: roomFinished ? 'room_finished' : 'timeout' };
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
