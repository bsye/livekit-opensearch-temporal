import { defineSearchAttributeKey, SearchAttributeType } from '@temporalio/common'
import { defineQuery } from '@temporalio/workflow'
import type { ActionDecision, LiveKitEvent, RoomSignal, RoomState, ToolStep, Turn, TurnTiming } from './types.js'

export const TASK_QUEUE = 'livekit-rooms'

export const RoomName = defineSearchAttributeKey('RoomName', SearchAttributeType.KEYWORD)
export const ParticipantIdentities = defineSearchAttributeKey('ParticipantIdentities', SearchAttributeType.KEYWORD_LIST)

export const roomState = defineQuery<RoomState>('roomState')

export function isRoomSignal(payload: unknown): payload is RoomSignal {
  const type = (payload as RoomSignal | undefined)?.type
  return type === 'livekitEvent' || type === 'turn'
}

export function actorIcon(kind: string | undefined): string {
  switch (kind) {
    case 'AGENT':
      return '🤖'
    case 'INGRESS':
      return '📥'
    case 'EGRESS':
      return '📤'
    case 'SIP':
      return '☎️'
    default:
      return '👤'
  }
}

export function eventLabel(e: LiveKitEvent): string {
  if (e.participant) {
    const track = e.track ? ` (${e.track.type ?? 'AUDIO'})` : ''
    return `${actorIcon(e.participant.kind)} ${e.participant.identity} · ${e.event}${track}`
  }
  if (e.egressInfo) return `📤 egress · ${e.event}`
  if (e.ingressInfo) return `📥 ingress · ${e.event}`
  return `🏠 ${e.event}`
}

export function signalLabel(s: RoomSignal): string {
  return s.type === 'livekitEvent' ? eventLabel(s.data) : '💬 turn received'
}

export function turnLabel(t: Turn): string {
  const user = t.userText ? `👤 “${truncate(t.userText, 40)}” → ` : ''
  const marks = [
    t.memory?.text ? '🧠' : '',
    ...t.tools.map((s) => `🛠 ${s.name}`),
    t.actions.some((a) => a.decision === 'flagged') ? '⚠️' : '',
    t.interrupted ? '✂️ interrupted' : '',
    t.timing.voiceToVoiceMs !== undefined ? `⏱ ${seconds(t.timing.voiceToVoiceMs)}` : '',
  ].filter(Boolean)
  return truncate(
    `💬 ${t.index ?? ''} · ${user}🤖 “${truncate(t.reply, 40)}”${marks.length ? ` · ${marks.join(' · ')}` : ''}`,
    190,
  )
}

export function toolLabel(s: ToolStep): string {
  const args = Object.values(s.args)
    .map((v) => (typeof v === 'string' ? v : JSON.stringify(v)))
    .join(', ')
  const took = s.durationMs !== undefined ? ` · ${seconds(s.durationMs)}` : ''
  return truncate(
    `🛠 ${s.name}(${truncate(args, 50)})${took}${s.output ? ` → ${s.isError ? '❌ ' : ''}${s.output}` : ''}`,
    190,
  )
}

export function userTimingLabel(t: TurnTiming): string | undefined {
  const parts = [
    t.speechMs !== undefined ? `you spoke ${seconds(t.speechMs)}` : '',
    t.transcriptionMs !== undefined ? `recognised in ${seconds(t.transcriptionMs)}` : '',
    t.endOfTurnMs !== undefined ? `end of turn ${seconds(t.endOfTurnMs)}` : '',
  ].filter(Boolean)
  return parts.length ? `🎙 ${parts.join(' · ')}` : undefined
}

export function agentTimingLabel(t: TurnTiming): string | undefined {
  const parts = [
    t.firstTokenMs !== undefined ? `first token ${seconds(t.firstTokenMs)}` : '',
    t.firstAudioMs !== undefined ? `first audio ${seconds(t.firstAudioMs)}` : '',
    t.voiceToVoiceMs !== undefined ? `⏱ voice-to-voice ${seconds(t.voiceToVoiceMs)}` : '',
  ].filter(Boolean)
  return parts.length ? `🤖 ${parts.join(' · ')}` : undefined
}

export function seconds(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

export function actionLabel(a: ActionDecision): string {
  const why = a.reasons.length ? ` (${a.reasons.join('; ')})` : ''
  if (a.decision === 'audited') return `🔎 Laya audit · ${a.tool} ok ${a.intent?.toFixed(2) ?? ''}`
  if (a.decision === 'flagged') return `⚠️ Laya audit · ${a.tool} flagged${why}`
  return `🛡 approval · ${a.tool} → ${a.decision}${why}`
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}
