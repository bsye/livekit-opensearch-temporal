/**
 * Instead of routing outside the model: does Qwen3-Omni, given the agent's real tools, decide by itself
 * to call recall for memory turns, an action tool for actions, and nothing for chat? Same 220 turns
 * as run.ts, spoken by Kokoro and transcribed by Parakeet as in a live call. Input modes:
 *   audio        the model hears the turn (as the live omni agent does)
 *   audio+text   it hears the turn and also reads the transcript
 *   text         it reads the transcript only
 *
 * Prompt variants: plain (the agent's instructions, recall left to the model), example (plus one
 * worked recall exchange for a fact, the usual fix for a model that won't call a tool) and examples
 * (plus a second one: a recommendation that checks the user's preferences first).
 *
 *   npm run omni -w @bench/router -- [--modes audio,audio+text,text] [--prompt plain|example] [--limit N]
 */
import { appendFileSync, mkdirSync } from 'node:fs';
import { exampleMessages, instructions } from '@voice/agent/omni-prompt';
import { toolSpecs } from '@voice/agent/tool-specs';
import { dataPath, env } from '@voice/config';
import { speak } from './speech.js';
import { routerTurns, type Label, type Turn } from './turns.js';

const arg = (name: string, def: string) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : def);
const MODES = arg('--modes', 'audio,audio+text,text').split(',');
const LIMIT = Number(arg('--limit', '1000')); // per label
const PROMPT = arg('--prompt', 'plain');
const OMNI = env('OMNI_BASE_URL');
const ACTION_TOOLS = new Set(['set_reminder', 'cancel_reminder', 'send_email']);

type Decision = 'recall' | 'action' | 'none';
interface Row extends Turn {
  mode: string;
  transcript: string;
  tools: string[];
  decision: Decision;
  reply: string;
  ms: number;
}

const tools = toolSpecs();
const EXAMPLES = exampleMessages(); // the agent's own: a fact, then a preference
const system = instructions(new Date());

function wav(pcm: Int16Array, rate = 16_000): Blob {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcm.byteLength, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcm.byteLength, 40);
  return new Blob([Uint8Array.from(header), new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength).slice()], { type: 'audio/wav' });
}

async function transcribe(pcm: Int16Array): Promise<string> {
  const form = new FormData();
  form.append('file', wav(pcm), 'turn.wav');
  form.append('model', env('STT_MODEL'));
  const res = await fetch(`${env('SPEECH_BASE_URL')}/audio/transcriptions`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`stt: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { text: string }).text.trim();
}

async function uploadAudio(pcm: Int16Array): Promise<string> {
  const res = await fetch(`${OMNI}/audio/turns?rate=16000`, {
    method: 'POST',
    headers: { 'content-type': 'application/octet-stream' },
    body: new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength).slice(),
  });
  if (!res.ok) throw new Error(`upload: ${res.status}`);
  return ((await res.json()) as { id: string }).id;
}

async function decide(mode: string, transcript: string, audioId: string) {
  const content = mode === 'text' ? transcript : `${transcript} <audio:${audioId}>`;
  const started = Date.now();
  const res = await fetch(`${OMNI}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: env('OMNI_MODEL'),
      temperature: 0,
      max_tokens: 120,
      tools,
      transcript_with_audio: mode === 'audio+text',
      messages: [{ role: 'system', content: system }, ...(PROMPT === 'example' ? EXAMPLES.slice(0, 4) : PROMPT === 'examples' ? EXAMPLES : []), { role: 'user', content }],
    }),
  });
  if (!res.ok) throw new Error(`omni: ${res.status} ${await res.text()}`);
  const message = ((await res.json()) as { choices: { message: { content: string | null; tool_calls?: { function: { name: string } }[] } }[] })
    .choices[0].message;
  const called = (message.tool_calls ?? []).map((c) => c.function.name);
  const decision: Decision = called.includes('recall') ? 'recall' : called.some((t) => ACTION_TOOLS.has(t)) ? 'action' : 'none';
  return { tools: called, decision, reply: message.content ?? '', ms: Date.now() - started };
}

const byLabel = (label: Label) => routerTurns().filter((t) => t.label === label).slice(0, LIMIT);
const turns = [...byLabel('past'), ...byLabel('action'), ...byLabel('chat')];
const outDir = dataPath('benchmarks', 'router');
mkdirSync(outDir, { recursive: true });
const log = `${outDir}/omni-${PROMPT}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '')}.jsonl`;
console.log(`${turns.length} turns × ${MODES.join(', ')}, prompt ${PROMPT} → ${log}`);

const rows: Row[] = [];
for (const turn of turns) {
  const pcm = await speak(turn.text);
  const [transcript, audioId] = await Promise.all([transcribe(pcm), uploadAudio(pcm)]);
  for (const mode of MODES) {
    const row: Row = { ...turn, mode, transcript, ...(await decide(mode, transcript, audioId)) };
    rows.push(row);
    appendFileSync(log, `${JSON.stringify(row)}\n`);
  }
  process.stdout.write('.');
}

const pct = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : '-');
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;
console.log('\n\nmode        | memory turns → recall | action turns → action tool | chat turns → no tool | non-memory turns → recall | p50');
for (const mode of MODES) {
  const rs = rows.filter((r) => r.mode === mode);
  const of = (label: Label) => rs.filter((r) => r.label === label);
  const share = (label: Label, d: Decision) => pct(of(label).filter((r) => r.decision === d).length, of(label).length);
  const others = rs.filter((r) => r.label !== 'past');
  console.log(
    `${mode.padEnd(11)} | ${share('past', 'recall').padStart(21)} | ${share('action', 'action').padStart(26)} | ${share('chat', 'none').padStart(20)} | ${pct(others.filter((r) => r.decision === 'recall').length, others.length).padStart(25)} | ${median(rs.map((r) => r.ms))} ms`,
  );
}
console.log('reference   | Laya 3-way choice: 71% of memory turns, 14% of other turns; Laya OR (1st person AND match >= 4): 84% / 21%');
