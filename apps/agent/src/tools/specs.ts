import { llm } from '@livekit/agents';
import type { Actions } from './actions.js';
import { createTools } from './index.js';

/** The cascade/omni tools as OpenAI function specs, exactly as the agent offers them (for benchmarks). */
export function toolSpecs() {
  const noop = async () => '';
  const actions = { setReminder: noop, cancelReminder: noop, sendEmail: noop } as unknown as Actions;
  const tools = createTools(actions, { room: () => ({ sid: '' }), agent: () => '', reportAction: () => {}, onRecall: () => {} });
  return Object.entries(tools).map(([name, tool]) => ({
    type: 'function' as const,
    function: { name, description: tool.description, parameters: llm.toJsonSchema(tool.parameters) },
  }));
}
