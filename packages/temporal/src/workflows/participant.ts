import { condition, setDefaultSignalHandler, setHandler, sleep } from '@temporalio/workflow';
import { isRoomSignal, participantState } from '../definitions.js';
import type { ParticipantSessionState } from '../types.js';

// Safety net if room_finished / participant_left is never delivered (webhooks aren't guaranteed).
export const MAX_SESSION = '24 hours';
// The agent's last transcript and metrics arrive after the user has left.
const LATE_EVENT_GRACE = '10 seconds';

/** One participant's lane in the room's timeline; completes shortly after they leave. */
export async function participantSession(init: {
  roomSid: string;
  identity: string;
  kind?: string;
}): Promise<ParticipantSessionState & { endReason: string }> {
  const state: ParticipantSessionState = { ...init, sid: '', tracks: {}, transcript: [], metrics: [], actions: [] };
  let left = false;

  setDefaultSignalHandler((_label, signal) => {
    if (!isRoomSignal(signal)) return;
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
      case 'action':
        state.actions.push(signal.data);
        break;
    }
  });
  setHandler(participantState, () => state);

  const hasLeft = await condition(() => left, MAX_SESSION, { summary: '⏳ in the room (24h max)' });
  if (hasLeft) await sleep(LATE_EVENT_GRACE, { summary: '⏳ grace period for late events' });
  return { ...state, endReason: hasLeft ? 'left' : 'timeout' };
}
