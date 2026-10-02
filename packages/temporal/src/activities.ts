import { indexExchange, type Exchange } from '@voice/memory';

export async function indexConversationExchange(exchange: Exchange): Promise<{ topic: string; hasTask: boolean }> {
  const doc = await indexExchange(exchange);
  return { topic: doc.topic, hasTask: doc.hasTask };
}
