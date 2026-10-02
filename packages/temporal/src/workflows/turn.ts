import { proxyActivities } from '@temporalio/workflow'
import type * as activities from '../activities.js'
import { actionLabel, agentTimingLabel, seconds, toolLabel, truncate, userTimingLabel } from '../definitions.js'
import type { Turn } from '../types.js'

const { noteStep } = proxyActivities<typeof activities>({ startToCloseTimeout: '10 seconds' })
const { indexConversationExchange } = proxyActivities<typeof activities>({
  startToCloseTimeout: '30 seconds',
  retry: { maximumAttempts: 5 },
})

/**
 * One exchange of a room, as a child of its roomSession. Its rows, in order: the user's side (speaking,
 * recognition), the memory lookup, each tool call / approval / Laya audit, the agent's side (first
 * token, first audio, voice-to-voice), then indexing the exchange into conversation memory.
 */
export async function conversationTurn(turn: Turn & { roomSid: string; roomName?: string }): Promise<Turn> {
  const note = (summary: string) => noteStep.executeWithOptions({ startToCloseTimeout: '10 seconds', summary }, [])

  const user = turn.userText ? userTimingLabel(turn.timing) : undefined
  if (user) await note(user)
  if (turn.memory) {
    const what = turn.memory.text ? `“${truncate(turn.memory.text, 110)}”` : 'not used'
    const took = turn.memory.ms !== undefined ? ` · ${seconds(turn.memory.ms)}` : ''
    await note(`🧠 memory · ${turn.memory.route}${took} → ${what}`)
  }
  // tool steps and decisions in the order they happened
  const steps = [
    ...turn.tools.map((t) => ({ at: t.at, label: toolLabel(t) })),
    ...turn.actions.map((a) => ({ at: a.at, label: actionLabel(a) })),
  ].sort((a, b) => a.at - b.at)
  for (const step of steps) await note(step.label)
  const agent = agentTimingLabel(turn.timing)
  if (agent) await note(agent)

  if (turn.userText && turn.reply) {
    await indexConversationExchange
      .executeWithOptions(
        {
          startToCloseTimeout: '30 seconds',
          retry: { maximumAttempts: 5 },
          summary: `📚 remember: “${truncate(turn.userText, 60)}”`,
        },
        [
          {
            roomSid: turn.roomSid,
            roomName: turn.roomName,
            user: turn.user,
            agent: turn.agent,
            userText: turn.userText,
            agentText: turn.reply,
            startedAt: turn.startedAt,
            endedAt: turn.endedAt,
            fromMemory: turn.fromMemory,
          },
        ],
      )
      .catch(() => undefined) // a failed index must not fail the turn
  }
  return turn
}
