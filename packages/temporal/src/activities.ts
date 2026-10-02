import { indexExchange, type Exchange } from '@voice/memory';

export async function indexConversationExchange(exchange: Exchange): Promise<{ topic: string; hasTask: boolean }> {
  const doc = await indexExchange(exchange);
  return { topic: doc.topic, hasTask: doc.hasTask };
}

/**
 * Does nothing: a step of a turn (a tool call, an approval, a memory lookup) recorded as a labelled
 * activity, so it shows as its own row in the turn's timeline. The work itself happened in the agent.
 */
export async function noteStep(): Promise<void> {}
