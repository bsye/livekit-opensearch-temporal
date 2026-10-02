import { voice } from '@livekit/agents'
import { type RecallResult, recall } from '@voice/memory'

export class RecallPrefetch {
  private heard: string[] = []
  private pending?: { question: string; result: Promise<RecallResult> }

  constructor(private readonly excludeRoomSid: () => string) {}

  attach(session: voice.AgentSession): void {
    session.on(voice.AgentSessionEventTypes.UserInputTranscribed, ({ transcript, isFinal }) => {
      if (isFinal && transcript.trim()) this.add(transcript.trim())
    })
    session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item }) => {
      if (item.type === 'message' && item.role === 'assistant') this.clear()
    })
  }

  current(): { question: string; result: Promise<RecallResult> } | undefined {
    return this.pending
  }

  private add(text: string): void {
    this.heard.push(text)
    const question = this.heard.join(' ')
    const result = recall({ question, excludeRoomSid: this.excludeRoomSid() })
    result.catch(() => undefined)
    this.pending = { question, result }
  }

  private clear(): void {
    this.heard = []
    this.pending = undefined
  }
}
