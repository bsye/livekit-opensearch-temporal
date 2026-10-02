import { defineSearchAttributeKey, SearchAttributeType } from '@temporalio/common';
import { defineQuery } from '@temporalio/workflow';
import type { ParticipantSessionState, RoomSignal, RoomState } from './types.js';

export const TASK_QUEUE = 'livekit-rooms';

// Created by infra/temporal/setup.sh
export const RoomName = defineSearchAttributeKey('RoomName', SearchAttributeType.KEYWORD);
export const ParticipantIdentities = defineSearchAttributeKey('ParticipantIdentities', SearchAttributeType.KEYWORD_LIST);

export const roomState = defineQuery<RoomState>('roomState');
export const participantState = defineQuery<ParticipantSessionState>('participantState');

export function isRoomSignal(payload: unknown): payload is RoomSignal {
  const type = (payload as RoomSignal | undefined)?.type;
  return type === 'livekitEvent' || type === 'transcript' || type === 'agentMetrics' || type === 'action';
}

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

/**
 * The Temporal UI labels signals by name only, so each signal is sent under a readable name
 * (`👤 dalbi · track_published (AUDIO)`) and workflows accept any name via a default handler.
 */
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
      return `${t.role === 'assistant' ? '🤖' : '👤'} ${t.participant}: “${truncate(t.text, 60)}”${t.interrupted ? ' (interrupted)' : ''}`;
    }
    case 'agentMetrics': {
      const m = s.data;
      const ms = (v: number | undefined) => `${Math.round(v ?? 0)}ms`;
      const what = {
        stt_metrics: `stt ${ms(m.durationMs)}`,
        eou_metrics: `end of turn ${ms(m.endOfUtteranceDelayMs)}`,
        llm_metrics: `llm first token ${ms(m.ttftMs)}`,
        tts_metrics: `tts first audio ${ms(m.ttfbMs)}`,
        turn_latency: `⏱ voice-to-voice ${ms(m.durationMs)}`,
      }[m.type];
      return `🤖 ${m.participant ?? 'agent'} · ${what}`;
    }
    case 'action': {
      const a = s.data;
      const why = a.reasons.length ? ` (${a.reasons.join('; ')})` : '';
      if (a.decision === 'audited') return `🔎 ${a.participant} · audit ${a.tool} ok ${a.intent?.toFixed(2)}`;
      if (a.decision === 'flagged') return `⚠️ ${a.participant} · audit ${a.tool} flagged${why}`;
      return `🛡 ${a.participant} · approval ${a.tool} → ${a.decision}${why}`;
    }
  }
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 3)}…` : text;
}
