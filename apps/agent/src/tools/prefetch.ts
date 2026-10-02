import { voice } from '@livekit/agents'
import { type RecallResult, recall } from '@voice/memory'

/**
 * Runs recall on the user's words as soon as speech-to-text finalises them, while the turn detector
 * is still deciding whether they're done (~0.7 s). If the model then calls recall, the result is
 * already there; if it doesn't, the result is dropped when the agent replies. Whether to use memory
 * stays the model's decision: this only moves the search off the critical path.
 */
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

  /** This turn's prefetched search (possibly still running), if any. */
  current(): { question: string; result: Promise<RecallResult> } | undefined {
    return this.pending
  }

  private add(text: string): void {
    // a pause can split one question into segments: search everything said since the last reply
    this.heard.push(text)
    const question = this.heard.join(' ')
    const result = recall({ question, excludeRoomSid: this.excludeRoomSid() })
    result.catch(() => undefined) // surfaced if the tool uses it
    this.pending = { question, result }
  }

  private clear(): void {
    this.heard = []
    this.pending = undefined
  }
}
