// Speech-to-speech (NVIDIA NemotronLabs VoiceChat) vs the cascade (Gemma 26B), on the same turns,
// tools and memory:
//   action  50 spoken requests for a tool (router bench)   → did it call a tool?
//   chat    50 spoken turns that need no tool              → did it stay out of tools?
//   memory  LongMemEval library questions (loaded memory)  → did it recall, and answer right?
// Modes:
//   s2s       VoiceChat hears the turn, calls tools itself on its function channel
//   s2s+laya  as s2s, plus Laya routes the model's own live transcript at end of speech and, for
//             memory turns, runs recall and forces the call + result in before the model decides
//   cascade   Gemma 26B (LM Studio) on the turn text with the same tools (perfect transcript, no audio)
// Tool results are identical across modes. Memory answers are judged by Gemma with LongMemEval's prompt.
//
//   npm run bench -w voicechat-bench -- [--sets action,chat,memory] [--modes s2s,s2s+laya,cascade] [--limit N] [--verbose]
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { recallBrief } from 'livekit-temporal/memory';
import { warmReranker } from 'livekit-temporal/rerank';
import { ACTION, CHAT } from '../../router/src/turns.js';
import { MEMORY_RULES, NVIDIA_SYSTEM_MESSAGE, VOICECHAT_TOOLS } from '../../../agents/voice-agent/src/s2s_prompt.js';
import { speak } from './speech.js';
import { ascii, FRAME_MS, FRAME_SAMPLES, parseToolCalls, systemPrompt, type ToolCall, type ToolSpec, VoiceChatSession } from '../../../agents/voice-agent/src/voicechat.js';

const VOICECHAT = process.env.VOICECHAT_URL ?? 'ws://localhost:8200';
const LLM = process.env.LLM_BASE_URL ?? 'http://localhost:1234/v1';
const LAYA = process.env.LAYA_BASE_URL ?? 'http://localhost:8100';
const arg = (name: string, def: string) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : def);
const SETS = arg('--sets', 'action,chat,memory').split(',');
const MODES = arg('--modes', 's2s,s2s+laya,cascade').split(',');
const LIMIT = Number(arg('--limit', '1000'));
const VERBOSE = process.argv.includes('--verbose');
const OUT = fileURLToPath(new URL('../../../data/benchmarks/voicechat/', import.meta.url));

const LEAD_SILENCE_FRAMES = 6; // 0.5 s before the user speaks
const QUIET_AFTER_REPLY_MS = 1500; // reply over: no new reply token for this long
const MAX_AFTER_SPEECH_MS = 20_000;

const TOOLS: ToolSpec[] = VOICECHAT_TOOLS;

const INSTRUCTIONS = (now: Date) =>
  'You are a helpful voice assistant. Keep replies short and conversational: one or two sentences. ' +
  'Use tools only when the user asks for that action. After a tool runs, tell the user what it did. ' +
  'Whenever the user asks about their own life, past, plans, purchases, people they know, or anything they told you ' +
  'before, call recall first. Answer from what recall returns and say when it was; if the user said different things ' +
  'at different times, the most recent one is current; if it does not answer the question, say you do not remember. ' +
  `Today is ${now.toDateString()}.`;

// ---- cases
function groupBy<T>(xs: T[], key: (x: T) => string): Record<string, T[]> {
  const out: Record<string, T[]> = {};
  for (const x of xs) (out[key(x)] ??= []).push(x);
  return out;
}
interface Case {
  set: 'action' | 'chat' | 'memory';
  text: string;
  type?: string; // LongMemEval question type
  answer?: string;
}
const sheet = readFileSync(new URL('../../../data/benchmarks/longmemeval/library-questions.md', import.meta.url), 'utf8');
const memory: Case[] = sheet
  .split('\n')
  .filter((l) => l.startsWith('| ') && !l.startsWith('| type'))
  .map((l) => l.split(' | '))
  .map((c) => ({ set: 'memory' as const, type: c[0].replace('| ', ''), text: c[2].trim(), answer: c[3].replace(/ \|$/, '').trim() }));
// spread the limit over question types
const byType = groupBy(memory, (c) => c.type!);
const memoryCases: Case[] = [];
for (let i = 0; memoryCases.length < Math.min(LIMIT, memory.length); i++)
  for (const list of Object.values(byType)) if (list[i] && memoryCases.length < LIMIT) memoryCases.push(list[i]);
const cases: Case[] = [
  ...(SETS.includes('action') ? ACTION.slice(0, LIMIT).map((text) => ({ set: 'action' as const, text })) : []),
  ...(SETS.includes('chat') ? CHAT.slice(0, LIMIT).map((text) => ({ set: 'chat' as const, text })) : []),
  ...(SETS.includes('memory') ? memoryCases : []),
];

// ---- tools (same results for every mode)
async function runTool(call: ToolCall): Promise<string> {
  const a = call.arguments as Record<string, string>;
  switch (call.name) {
    case 'set_reminder':
      return `Reminder set: ${a.text} at ${a.time} ${a.day}.`;
    case 'cancel_reminder':
      return 'Cancelled the most recent reminder.';
    case 'send_email':
      return (call.arguments.confirmed ? `Email to ${a.to} sent.` : `Not sent yet: read it back and ask the user to confirm.`);
    case 'recall':
      return recallForVoice(a.question ?? '');
    default:
      return `Unknown tool ${call.name}.`;
  }
}

const recallForVoice = async (question: string) => (await recallBrief({ question })).text;

// ---- Laya router (the 3-way choice from benchmarks/router)
async function layaRoute(text: string): Promise<'past' | 'action' | 'chat'> {
  const res = await fetch(`${LAYA}/v1/systemone`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      state: `User: ${text}`,
      questions: {
        q: {
          type: 'choice',
          instructions: 'What does the user want from the assistant in this turn?',
          criteria: {
            A: 'Information from their own past or from earlier conversations with the assistant',
            B: 'An action: a reminder, timer, message, email, calendar entry, call or other task',
            C: 'Conversation, general knowledge, advice or anything else',
          },
        },
      },
    }),
  });
  const choice = ((await res.json()) as { answers: { q: { choice: string } } }).answers.q.choice;
  return ({ A: 'past', B: 'action', C: 'chat' } as const)[choice as 'A' | 'B' | 'C'] ?? 'chat';
}

// ---- one turn through VoiceChat
interface Result {
  mode: string;
  set: string;
  type?: string;
  text: string;
  heard?: string; // the model's transcript of the user
  functionText?: string; // raw function channel
  reply: string;
  tools: string[];
  routed?: string; // Laya's route (s2s+laya)
  prefetched?: boolean;
  replyMs?: number; // end of user speech → first reply token
  toolMs?: number; // end of user speech → tool call complete (or prefetch injected)
  injectMs?: number; // compute to force the result in
  answerMs?: number; // end of user speech → first reply token after the tool result
  correct?: boolean;
  frameP50?: number;
  frameP95?: number;
  error?: string;
}

async function s2sTurn(c: Case, withLaya: boolean): Promise<Result> {
  const audio = await speak(c.text);
  const s = await VoiceChatSession.open(VOICECHAT, systemPrompt(`${NVIDIA_SYSTEM_MESSAGE}\n\n${MEMORY_RULES} Today is ${new Date().toDateString()}.`, TOOLS));
  const r: Result = { mode: withLaya ? 's2s+laya' : 's2s', set: c.set, type: c.type, text: c.text, reply: '', tools: [] };
  const frames: Int16Array[] = [];
  for (let i = 0; i < LEAD_SILENCE_FRAMES; i++) frames.push(new Int16Array(FRAME_SAMPLES));
  for (let i = 0; i < audio.length; i += FRAME_SAMPLES) {
    const f = new Int16Array(FRAME_SAMPLES);
    f.set(audio.subarray(i, i + FRAME_SAMPLES));
    frames.push(f);
  }
  const speechFrames = frames.length;
  let speechEndAt = Infinity;
  let handled = 0;
  let pending: Promise<void> = Promise.resolve();
  let resultInAt = 0; // wall clock when the last tool result finished going in
  const toolText = () => s.functionText.replace(/<TOOL_RESPONSE>[\s\S]*?<\/TOOL_RESPONSE>/g, '');

  // the model's own tool calls: run them and return the result as soon as the call closes
  s.onFunction = () => {
    const calls = parseToolCalls(toolText());
    for (; handled < calls.length; handled++) {
      const call = calls[handled];
      r.tools.push(call.name);
      r.toolMs ??= Date.now() - speechEndAt;
      pending = pending.then(async () => {
        const out = await runTool(call);
        const t = Date.now();
        await s.toolOutput(out);
        r.injectMs = Date.now() - t;
        resultInAt = Date.now();
      });
    }
  };

  // real-time: one 80 ms frame per 80 ms of wall clock
  const start = Date.now();
  let i = 0;
  const sendUntil = async (done: () => boolean) => {
    for (; !done(); i++) {
      const wait = start + i * FRAME_MS - Date.now();
      if (wait > 0) await new Promise((res) => setTimeout(res, wait));
      s.push(i < frames.length ? frames[i] : new Int16Array(FRAME_SAMPLES));
      if (i === speechFrames - 1) {
        speechEndAt = Date.now();
        if (withLaya) {
          // route on what the model heard; prefetch memory before it decides
          const heard = s.userText;
          pending = pending.then(async () => {
            r.routed = await layaRoute(heard);
            if (r.routed !== 'past' || parseToolCalls(toolText()).length) return;
            const call = { name: 'recall', arguments: { question: heard } };
            const out = await runTool(call);
            if (parseToolCalls(toolText()).length) return; // the model called a tool meanwhile
            handled++; // the forced call appears on the function channel: don't run it again
            r.prefetched = true;
            r.tools.push('recall (prefetch)');
            r.toolMs = Date.now() - speechEndAt;
            const t = Date.now();
            await s.toolOutput(out, call);
            r.injectMs = Date.now() - t;
            resultInAt = Date.now();
          });
        }
      }
    }
  };
  const lastReplyAt = () => s.textEvents.at(-1)?.at ?? 0;
  await sendUntil(() => {
    if (i < speechFrames) return false;
    const now = Date.now();
    if (now - speechEndAt > MAX_AFTER_SPEECH_MS) return true;
    const replied = s.textEvents.some((e) => e.at > Math.max(speechEndAt, resultInAt));
    const toolsBusy = parseToolCalls(toolText()).length > (resultInAt ? handled : 0) || (!!r.tools.length && !resultInAt);
    return replied && !toolsBusy && now - lastReplyAt() > QUIET_AFTER_REPLY_MS;
  });
  await pending;
  await s.drain();
  const st = await s.stats();
  s.close();
  r.heard = s.userText;
  r.reply = s.assistantText;
  r.functionText = s.functionText;
  const firstAfter = (t: number) => s.textEvents.find((e) => e.at > t)?.at;
  r.replyMs = (firstAfter(speechEndAt) ?? NaN) - speechEndAt;
  if (resultInAt) r.answerMs = (firstAfter(resultInAt) ?? NaN) - speechEndAt;
  r.frameP50 = st.frame_ms_p50;
  r.frameP95 = st.frame_ms_p95;
  return r;
}

// ---- the cascade's LLM on the same turn (text in, same tools and results)
async function cascadeTurn(c: Case): Promise<Result> {
  const r: Result = { mode: 'cascade', set: c.set, type: c.type, text: c.text, reply: '', tools: [] };
  const messages: Record<string, unknown>[] = [
    { role: 'system', content: INSTRUCTIONS(new Date()) },
    { role: 'user', content: c.text },
  ];
  const t0 = Date.now();
  for (let step = 0; step < 3; step++) {
    const res = await fetch(`${LLM}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: process.env.LLM_MODEL,
        reasoning_effort: 'none',
        temperature: 0,
        messages,
        tools: TOOLS.map((t) => ({ type: 'function', function: t })),
      }),
    });
    const msg = ((await res.json()) as { choices: { message: Record<string, unknown> }[] }).choices[0].message;
    const calls = (msg.tool_calls ?? []) as { id: string; function: { name: string; arguments: string } }[];
    if (!calls.length) {
      r.reply = String(msg.content ?? '').replace(/<\|channel>[\s\S]*?<channel\|>/g, '').trim();
      r.answerMs = Date.now() - t0;
      break;
    }
    messages.push(msg);
    for (const call of calls) {
      r.tools.push(call.function.name);
      r.toolMs ??= Date.now() - t0;
      const out = await runTool({ name: call.function.name, arguments: JSON.parse(call.function.arguments || '{}') });
      messages.push({ role: 'tool', tool_call_id: call.id, content: ascii(out) });
    }
  }
  return r;
}

// ---- LongMemEval's answer judge (its default prompt), with Gemma
async function judge(c: Case, reply: string): Promise<boolean> {
  const prompt =
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response ' +
    'contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains ' +
    'all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a ' +
    'subset of the information required by the answer, answer no. \n\n' +
    `Question: ${c.text}\n\nCorrect Answer: ${c.answer}\n\nModel Response: ${reply}\n\n` +
    'Is the model response correct? Answer yes or no only.';
  const res = await fetch(`${LLM}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: process.env.LLM_MODEL, reasoning_effort: 'none', temperature: 0, max_tokens: 5, messages: [{ role: 'user', content: prompt }] }),
  });
  const out = ((await res.json()) as { choices: { message: { content: string } }[] }).choices[0].message.content;
  return /yes/i.test(out);
}

// ---- run
mkdirSync(OUT, { recursive: true });
const log = `${OUT}results-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '')}.jsonl`;
await warmReranker();
console.log(`${cases.length} cases × ${MODES.join(', ')} → ${log}`);
const results: Result[] = [];
for (const c of cases) {
  for (const mode of MODES) {
    let r: Result;
    try {
      r = mode === 'cascade' ? await cascadeTurn(c) : await s2sTurn(c, mode === 's2s+laya');
      if (c.set === 'memory') r.correct = await judge(c, r.reply);
    } catch (err) {
      r = { mode, set: c.set, text: c.text, reply: '', tools: [], error: String(err) };
    }
    results.push(r);
    appendFileSync(log, `${JSON.stringify(r)}\n`);
    if (VERBOSE) {
      console.log(`\n[${mode}] ${c.set}${c.type ? `/${c.type}` : ''}: ${c.text}`);
      if (r.heard) console.log(`  heard:  ${r.heard}`);
      if (r.functionText) console.log(`  fn:     ${JSON.stringify(r.functionText)}`);
      console.log(`  tools:  ${r.tools.join(', ') || '-'}${r.routed ? `   (laya: ${r.routed})` : ''}`);
      console.log(`  reply:  ${r.reply}`);
      if (c.answer) console.log(`  expect: ${c.answer} → ${r.correct ? 'correct' : 'wrong'}`);
      console.log(`  ms: reply ${r.replyMs ?? '-'} · tool ${r.toolMs ?? '-'} · inject ${r.injectMs ?? '-'} · answer ${r.answerMs ?? '-'} · frame p50 ${r.frameP50?.toFixed(0) ?? '-'} p95 ${r.frameP95?.toFixed(0) ?? '-'}${r.error ? ` · ERROR ${r.error}` : ''}`);
    } else process.stdout.write(r.error ? 'x' : '.');
  }
}

// ---- summary
const pct = (n: number, d: number) => (d ? `${((100 * n) / d).toFixed(0)}%` : '-');
const med = (xs: (number | undefined)[]) => {
  const v = xs.filter((x): x is number => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? `${Math.round(v[Math.floor(v.length / 2)])}` : '-';
};
console.log('\n');
for (const mode of MODES) {
  const rs = results.filter((r) => r.mode === mode && !r.error);
  const of = (set: string) => rs.filter((r) => r.set === set);
  const acting = (r: Result) => r.tools.some((t) => !t.startsWith('recall'));
  const recalled = (r: Result) => r.tools.some((t) => t.startsWith('recall'));
  console.log(`== ${mode}  (${rs.length} turns, ${results.filter((r) => r.mode === mode && r.error).length} errors)`);
  if (of('action').length) console.log(`   action: tool called ${pct(of('action').filter(acting).length, of('action').length)}`);
  if (of('chat').length) console.log(`   chat:   no tool ${pct(of('chat').filter((r) => !r.tools.length).length, of('chat').length)}`);
  if (of('memory').length) {
    const m = of('memory');
    console.log(`   memory: recalled ${pct(m.filter(recalled).length, m.length)}, answered correctly ${pct(m.filter((r) => r.correct).length, m.length)}`);
    for (const [type, list] of Object.entries(groupBy(m, (r) => r.type ?? '')))
      console.log(`     ${type.padEnd(28)} ${pct(list.filter((r) => r.correct).length, list.length)} (n=${list.length})`);
  }
  console.log(`   ms p50: first reply ${med(rs.map((r) => r.replyMs))} · tool ${med(rs.map((r) => r.toolMs))} · inject ${med(rs.map((r) => r.injectMs))} · answer after tool ${med(rs.filter((r) => r.tools.length).map((r) => r.answerMs))}`);
  if (mode !== 'cascade') console.log(`   model step p50 ${med(rs.map((r) => r.frameP50))} ms, p95 ${med(rs.map((r) => r.frameP95))} ms (real time needs < ${FRAME_MS})`);
}
