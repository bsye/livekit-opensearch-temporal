import {
  condition,
  proxyActivities,
  setDefaultSignalHandler,
  setHandler,
  sleep,
  startChild,
  upsertSearchAttributes,
  workflowInfo,
  type ChildWorkflowHandle,
} from '@temporalio/workflow';
import type * as activities from '../activities.js';
import { actorIcon, isRoomSignal, ParticipantIdentities, RoomName, roomState, truncate } from '../definitions.js';
import type { LiveKitEvent, RoomSignal, RoomState, TranscriptEntry } from '../types.js';
import { MAX_SESSION, participantSession } from './participant.js';

// Webhooks can arrive after room_finished.
const LATE_EVENT_GRACE = '1 minute';

/**
 * One workflow per LiveKit room session (id = room sid). Holds the combined room state, starts a
 * participantSession child per participant (one lane each in the timeline), and indexes every
 * exchange into conversation memory.
 */
export async function roomSession(): Promise<RoomState & { endReason: string }> {
  const state: RoomState = {
    participants: {},
    egress: {},
    transcript: [],
    metrics: [],
    actions: [],
    eventCount: 0,
    duplicateCount: 0,
  };
  const seen = new Set<string>();
  const lanes = new ParticipantLanes(() => state.name);
  const memory = new MemoryIndexer(() => state.sid ?? workflowInfo().workflowId, () => state.name);
  let finished = false;

  setDefaultSignalHandler((label, signal) => {
    if (!isRoomSignal(signal)) return;
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
        if (e.participant) lanes.forward(e.participant.identity, e.participant.kind, signal, label);
        if (e.event === 'room_finished') finished = true;
        break;
      }
      case 'transcript':
        state.transcript.push(signal.data);
        lanes.forward(signal.data.participant, undefined, signal, label);
        memory.add(signal.data);
        break;
      case 'agentMetrics':
        state.metrics.push(signal.data);
        if (signal.data.participant) lanes.forward(signal.data.participant, 'AGENT', signal, label);
        break;
      case 'action':
        state.actions.push(signal.data);
        lanes.forward(signal.data.participant, 'AGENT', signal, label);
        break;
    }
  });
  setHandler(roomState, () => state);

  const roomFinished = await condition(() => finished, MAX_SESSION, { summary: '⏳ waiting for room_finished (24h max)' });
  if (roomFinished) await sleep(LATE_EVENT_GRACE, { summary: '⏳ grace period for late webhooks' });
  return { ...state, endReason: roomFinished ? 'room_finished' : 'timeout' };
}

/** Pairs the user's turn(s) with the agent's reply and indexes each pair as an activity. */
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
    if (this.userTurns.length === 0) return;
    const turns = this.userTurns;
    this.userTurns = [];
    const userText = turns.map((t) => t.text).join(' ');
    const { indexConversationExchange } = proxyActivities<typeof activities>({
      startToCloseTimeout: '30 seconds',
      retry: { maximumAttempts: 5 },
      summary: `🧠 remember: “${truncate(userText, 50)}”`,
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
      fromMemory: entry.fromMemory,
    }).catch(() => undefined); // a failed index must not fail the room session
  }
}

/** Starts one participantSession child per identity and forwards its signals in order. */
class ParticipantLanes {
  private lanes = new Map<string, Promise<ChildWorkflowHandle<typeof participantSession> | undefined>>();

  constructor(private roomName: () => string | undefined) {}

  forward(identity: string, kind: string | undefined, signal: RoomSignal, label: string): void {
    const previous = this.lanes.get(identity) ?? this.start(identity, kind);
    const next = previous.then(async (child) => {
      // a lane that already completed just drops late signals
      await child?.signal(label, signal).catch(() => undefined);
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
    }).catch(() => undefined);
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
