// Strips Gemma control tokens from the LLM's streamed text before it reaches TTS and the
// transcript. After a tool call, gemma-4 sometimes opens a thought channel in plain content
// ("<|channel>You mentioned…"), which Kokoro would read out. Any thought text inside
// "<|channel>thought … <channel|>" is dropped; stray markers are removed.
import type { llm } from '@livekit/agents';

const MARKERS = ['<|channel>', '<channel|>'];
const LONGEST = Math.max(...MARKERS.map((m) => m.length));

export async function* stripControlTokens(
  stream: AsyncIterable<llm.ChatChunk | string>,
): AsyncIterable<llm.ChatChunk | string> {
  let pending = ''; // text held back because it may be the start of a marker
  let inThought = false;

  const clean = (text: string, final: boolean): string => {
    pending += text;
    let out = '';
    for (;;) {
      if (inThought) {
        const end = pending.indexOf('<channel|>');
        if (end === -1) {
          pending = final ? '' : pending.slice(-LONGEST);
          return out;
        }
        pending = pending.slice(end + '<channel|>'.length);
        inThought = false;
        continue;
      }
      const start = pending.indexOf('<|channel>');
      if (start !== -1) {
        out += pending.slice(0, start);
        pending = pending.slice(start + '<|channel>'.length);
        if (pending.startsWith('thought')) inThought = true;
        continue;
      }
      pending = pending.replaceAll('<channel|>', '');
      // keep a tail that could still grow into a marker
      const keep = final ? 0 : partialMarkerLength(pending);
      out += pending.slice(0, pending.length - keep);
      pending = pending.slice(pending.length - keep);
      return out;
    }
  };

  for await (const chunk of stream) {
    if (typeof chunk === 'string') {
      const text = clean(chunk, false);
      if (text) yield text;
      continue;
    }
    if (chunk.delta?.content) {
      const content = clean(chunk.delta.content, false);
      yield { ...chunk, delta: { ...chunk.delta, content } };
    } else {
      yield chunk; // tool calls, usage: untouched
    }
  }
  const rest = clean('', true);
  if (rest) yield rest;
}

/** Length of the longest suffix of text that is a prefix of a marker. */
function partialMarkerLength(text: string): number {
  for (let n = Math.min(LONGEST - 1, text.length); n > 0; n--) {
    const tail = text.slice(-n);
    if (MARKERS.some((m) => m.startsWith(tail))) return n;
  }
  return 0;
}
