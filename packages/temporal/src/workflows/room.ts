import {
  condition,
  setCurrentDetails,
  setDefaultSignalHandler,
  setHandler,
  sleep,
  startChild,
  upsertSearchAttributes,
  workflowInfo,
} from '@temporalio/workflow'
import { tableCell } from '@voice/text'
import {
  actorIcon,
  isRoomSignal,
  ParticipantIdentities,
  RoomName,
  roomState,
  seconds,
  truncate,
  turnLabel,
} from '../definitions.js'
import type { LiveKitEvent, RoomState, Turn } from '../types.js'
import { conversationTurn } from './turn.js'

const MAX_SESSION = '24 hours'
const LATE_EVENT_GRACE = '1 minute'
const TURNS_IN_DETAILS = 12

export async function roomSession(): Promise<RoomState & { endReason: string }> {
  const state: RoomState = { participants: {}, egress: {}, turns: [], eventCount: 0, duplicateCount: 0 }
  const seen = new Set<string>()
  let finished = false
  let chain: Promise<unknown> = Promise.resolve()

  setDefaultSignalHandler((_label, signal) => {
    if (!isRoomSignal(signal)) return
    if (signal.type === 'livekitEvent') {
      const e = signal.data
      if (seen.has(e.id)) {
        state.duplicateCount++
        return
      }
      seen.add(e.id)
      state.eventCount++
      apply(state, e)
      if (e.event === 'room_finished') finished = true
    } else {
      const userTurns = state.turns.filter((t) => t.userText).length
      const turn: Turn = { ...signal.data, index: signal.data.userText ? userTurns + 1 : 0 }
      state.turns.push(turn)
      const n = state.turns.length
      const roomSid = state.sid ?? workflowInfo().workflowId
      chain = chain.then(() =>
        startChild(conversationTurn, {
          workflowId: `${workflowInfo().workflowId}/turn-${n}`,
          args: [{ ...turn, roomSid, roomName: state.name }],
          staticSummary: turnLabel(turn),
          staticDetails: turnDetails(turn),
          typedSearchAttributes: state.name ? [{ key: RoomName, value: state.name }] : [],
        }).catch(() => undefined),
      )
    }
    setCurrentDetails(details(state))
  })
  setHandler(roomState, () => state)

  const roomFinished = await condition(() => finished, MAX_SESSION, {
    summary: '⏳ waiting for room_finished (24h max)',
  })
  if (roomFinished) await sleep(LATE_EVENT_GRACE, { summary: '⏳ grace period for late events' })
  await chain
  return { ...state, endReason: roomFinished ? 'room_finished' : 'timeout' }
}

function details(state: RoomState): string {
  const people = Object.values(state.participants).map(
    (p) =>
      `| ${actorIcon(p.kind)} ${p.identity} | ${p.kind ?? 'STANDARD'} | ${Object.values(p.tracks)
        .filter((t) => !t.unpublishedAt)
        .map((t) => t.type ?? 'AUDIO')
        .join(', ')} | ${p.leftAt ? 'left' : 'in the room'} |`,
  )
  const turns = state.turns
    .slice(-TURNS_IN_DETAILS)
    .map(
      (t) =>
        `| ${t.index} | ${tableCell(truncate(t.userText, 80))} | ${tableCell(truncate(t.reply, 100))} | ${[
          t.memory?.text ? '🧠 memory' : '',
          ...t.tools.map((s) => `🛠 ${s.name}`),
          ...t.actions
            .filter((a) => a.decision !== 'audited')
            .map((a) => `${a.decision === 'flagged' ? '⚠️' : '🛡'} ${a.tool} ${a.decision}`),
        ]
          .filter(Boolean)
          .join(
            ', ',
          )} | ${ms(t.timing.endOfTurnMs)} | ${ms(t.timing.firstTokenMs)} | ${ms(t.timing.firstAudioMs)} | ${ms(t.timing.voiceToVoiceMs)} |`,
    )
  return [
    `### 🏠 ${state.name ?? ''}`,
    '',
    '| participant | kind | tracks | status |',
    '|---|---|---|---|',
    ...people,
    '',
    `### 💬 Turns (${state.turns.length}${state.turns.length > TURNS_IN_DETAILS ? `, last ${TURNS_IN_DETAILS}` : ''})`,
    '',
    '| # | user | agent | tools | 🎙 end of turn | 💭 first token | 🔊 first audio | ⏱ voice-to-voice |',
    '|---|---|---|---|---|---|---|---|',
    ...turns,
  ].join('\n')
}

const ms = (v: number | undefined) => (v === undefined ? '' : seconds(v))

function turnDetails(t: Turn): string {
  const row = (label: string, v: number | undefined) => (v === undefined ? [] : [`| ${label} | ${seconds(v)} |`])
  return [
    `**👤 ${t.user}:** ${t.userText || '—'}`,
    '',
    `**🤖 ${t.agent}:** ${t.reply}${t.interrupted ? ' *(interrupted)*' : ''}`,
    ...(t.memory?.text ? ['', `**🧠 memory (${t.memory.route}):** ${t.memory.text}`] : []),
    '',
    '| where the time went | |',
    '|---|---|',
    ...row('🎙 you spoke', t.timing.speechMs),
    ...row('📝 speech recognised in', t.timing.transcriptionMs),
    ...row('⏸ end of turn (after you stopped)', t.timing.endOfTurnMs),
    ...row('🧠 memory', t.memory?.ms),
    ...t.tools.map((s) => `| 🛠 ${s.name} | ${s.durationMs !== undefined ? seconds(s.durationMs) : ''} |`),
    ...row('💭 first token', t.timing.firstTokenMs),
    ...row('🔊 first audio', t.timing.firstAudioMs),
    ...row('⏱ voice-to-voice', t.timing.voiceToVoiceMs),
  ].join('\n')
}

function apply(state: RoomState, e: LiveKitEvent): void {
  if (e.room) {
    state.sid = e.room.sid
    state.name = e.room.name
  }
  switch (e.event) {
    case 'room_started':
      state.startedAt = e.createdAt
      break
    case 'room_finished':
      state.finishedAt = e.createdAt
      break
    case 'participant_joined': {
      const p = participant(state, e)
      if (!p) break
      p.joinedAt = e.createdAt
      upsertSearchAttributes([{ key: ParticipantIdentities, value: Object.keys(state.participants) }])
      break
    }
    case 'participant_left':
    case 'participant_connection_aborted': {
      const p = participant(state, e)
      if (p) p.leftAt = e.createdAt
      break
    }
    case 'track_published':
    case 'track_unpublished': {
      const p = participant(state, e)
      if (!p || !e.track) break
      const t = (p.tracks[e.track.sid] ??= { sid: e.track.sid })
      Object.assign(t, { type: e.track.type, source: e.track.source, mimeType: e.track.mimeType })
      if (e.event === 'track_published') t.publishedAt = e.createdAt
      else t.unpublishedAt = e.createdAt
      break
    }
    default:
      if (e.egressInfo) state.egress[e.egressInfo.egressId] = { status: e.egressInfo.status }
  }
}

function participant(state: RoomState, e: LiveKitEvent) {
  if (!e.participant) return undefined
  const { identity, sid, kind } = e.participant
  return (state.participants[identity] ??= { identity, sid, kind, tracks: {} })
}
