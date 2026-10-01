// Temporal activities (run by the worker, outside the workflow sandbox: network calls are fine).
import { indexExchange, type Exchange } from './memory.js';

/** Categorise an exchange with Laya and store it in conversation memory (OpenSearch). */
export async function indexConversationExchange(exchange: Exchange): Promise<{ topic: string; hasTask: boolean }> {
  const doc = await indexExchange(exchange);
  return { topic: doc.topic, hasTask: doc.hasTask };
}
