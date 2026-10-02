import type { llm } from '@livekit/agents';

const OPEN = '<|channel>';
const CLOSE = '<channel|>';
const MARKERS = [OPEN, CLOSE];
const LONGEST = Math.max(...MARKERS.map((m) => m.length));

/**
 * After a tool call gemma-4 sometimes opens a thought channel in plain content ("<|channel>thought …
 * <channel|>"), which TTS would read out. Drops the thought and stray markers from the stream.
 */
export async function* stripControlTokens(
  stream: AsyncIterable<llm.ChatChunk | string>,
): AsyncIterable<llm.ChatChunk | string> {
  let pending = '';
  let inThought = false;

  const clean = (text: string, final: boolean): string => {
    pending += text;
    let out = '';
    for (;;) {
      if (inThought) {
        const end = pending.indexOf(CLOSE);
        if (end === -1) {
          pending = final ? '' : pending.slice(-LONGEST);
          return out;
        }
        pending = pending.slice(end + CLOSE.length);
        inThought = false;
        continue;
      }
      const start = pending.indexOf(OPEN);
      if (start !== -1) {
        out += pending.slice(0, start);
        pending = pending.slice(start + OPEN.length);
        if (pending.startsWith('thought')) inThought = true;
        continue;
      }
      pending = pending.replaceAll(CLOSE, '');
      // hold back a tail that could still grow into a marker
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
    } else if (chunk.delta?.content) {
      yield { ...chunk, delta: { ...chunk.delta, content: clean(chunk.delta.content, false) } };
    } else {
      yield chunk;
    }
  }
  const rest = clean('', true);
  if (rest) yield rest;
}

function partialMarkerLength(text: string): number {
  for (let n = Math.min(LONGEST - 1, text.length); n > 0; n--) {
    const tail = text.slice(-n);
    if (MARKERS.some((m) => m.startsWith(tail))) return n;
  }
  return 0;
}
