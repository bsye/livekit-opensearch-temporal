// Speech-to-speech voice agent: NVIDIA NemotronLabs VoiceChat (services/voicechat, MLX) in place of
// the STT → LLM → TTS cascade (agent.ts). The model is full duplex: the user's audio streams in
// continuously and its own audio streams out every 80 ms, so it does its own turn-taking and
// barge-in. Tool calls arrive on a separate function channel and run through the same tools as the
// cascade (reminders as Temporal workflows, memory recall, email); Laya routes every user turn to
// decide when to search memory (the model rarely does it itself) and audits every committed action. Transcripts (the model's own) and voice-to-voice latency go to the room's Temporal workflow.
//
//   AGENT=s2s npm run meet   (or: npm run dev:s2s -w voice-agent)
import { fileURLToPath } from 'node:url';
import { cli, defineAgent, ServerOptions, type JobContext, type JobProcess } from '@livekit/agents';
import {
  AudioFrame,
  AudioSource,
  AudioStream,
  LocalAudioTrack,
  RoomEvent,
  TrackKind,
  TrackPublishOptions,
  TrackSource,
  type RemoteTrack,
} from '@livekit/rtc-node';
import { connectTemporal, recordEmail, signalRoom } from 'livekit-temporal/client';
import { recallBrief } from 'livekit-temporal/memory';
import { warmReranker } from 'livekit-temporal/rerank';
import type { GateDecision, RoomSignal } from 'livekit-temporal/shared';
import { ActionAuditor } from './audit.js';
import { prompt } from './s2s_prompt.js';
import { createTools } from './tools.js';
import { FRAME_SAMPLES, parseToolCalls, type ToolCall, VoiceChatSession } from './voicechat.js';

const OUTPUT_RATE = 22_050; // VoiceChat's speech
// Route the user's turn once it's over: when the model starts replying, or after this much quiet
// (transcript words arrive with gaps over 300 ms mid-sentence, so shorter routes half questions)
const PREFETCH_AFTER_MS = 700;
const FIRST_PERSON = /\b(i|my|me|mine|i'm|i've|i'd|i'll)\b/i;
const REFERS_BACK =
  /\b(previous|earlier|last time|before|remind me what|we (talked|discussed|spoke|said)|you (told|said|mentioned|recommended|suggested)|did i (tell|mention|say))\b/i;
const STRONG_MATCH = 4; // MiniLM logit
const MAX_HOLD_MS = 2500; // never hold the model's voice longer than this
const MAX_CALLS_PER_TURN = 3; // after that the model is told to answer (it can retry a failing call forever)
// A reply (or user turn) is over when its text hasn't grown for this long
const TURN_QUIET_MS = 1200;

export default defineAgent({
  prewarm: async (_proc: JobProcess) => {
    await warmReranker(); // memory recall's MiniLM, so the first recall isn't slow
    // the prompt itself is prefilled by src/warm_s2s.ts (meet.sh), not here: it takes ~1 min
  },

  entry: async (ctx: JobContext) => {
    const temporal = await connectTemporal();
    await ctx.connect();
    const room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' };
    const agentIdentity = ctx.room.localParticipant?.identity ?? 'agent';
    let user = 'user';
    const report = (signal: RoomSignal) =>
      signalRoom(temporal, room, signal).catch((err) => console.error(`temporal ${signal.type} signal failed`, err));
    const reportDecision = (decision: GateDecision) => report({ type: 'gate', data: decision });

    // ---- the model
    const t0 = Date.now();
    const model = await VoiceChatSession.open(env('VOICECHAT_URL'), prompt());
    console.log(`voicechat session ready in ${Date.now() - t0}ms`);

    // ---- tools: the cascade's, run on the model's calls
    let userText = ''; // the user's current turn, as the model heard it
    let answeringFromMemory = false;
    const auditor = new ActionAuditor(env('LAYA_BASE_URL'), () => agentIdentity, () => userText, reportDecision);
    const tools = createTools({
      temporal,
      room: () => room,
      user: () => user,
      agent: () => agentIdentity,
      auditor,
      report: reportDecision,
      onRecall: () => (answeringFromMemory = true),
    });
    // set_reminder / cancel_reminder don't use the cascade's run context (only send_email's approval step does)
    const runCascadeTool = (name: 'set_reminder' | 'cancel_reminder', args: Record<string, unknown>) =>
      (tools[name].execute as (a: unknown, o: unknown) => Promise<string>)(args, { ctx: undefined, toolCallId: name, abortSignal: new AbortController().signal });

    async function runTool(call: ToolCall): Promise<string> {
      const a = call.arguments as Record<string, string>;
      switch (call.name) {
        case 'set_reminder':
        case 'cancel_reminder':
          return runCascadeTool(call.name, a);
        case 'recall':
          answeringFromMemory = true;
          return (await recallBrief({ question: a.question ?? userText, excludeRoomSid: room.sid })).text;
        case 'send_email': {
          // the model asks the user and calls again once they said yes (VOICECHAT_TOOLS)
          const decision = call.arguments.confirmed === true ? 'confirmed' : 'confirm';
          reportDecision({ tool: 'send_email', args: call.arguments, decision, reasons: [], latencyMs: 0, at: Date.now(), participant: agentIdentity });
          if (decision === 'confirm') return 'Not sent yet. Read the email back to the user and ask them to confirm.';
          await recordEmail(temporal, { to: a.to, subject: a.subject, body: a.body, roomSid: room.sid, requestedBy: user }, room.name);
          auditor.audit('send_email', { to: a.to, subject: a.subject, body: a.body });
          return `Sent the email to ${a.to}.`;
        }
        default:
          return `There is no tool called ${call.name}.`;
      }
    }

    let handledCalls = 0;
    let toolQueue: Promise<void> = Promise.resolve();
    // loop guard: calls per user turn (reset when the user starts a new turn)
    let turnCalls = 0;
    let turnRecalled = false;
    model.onFunction = (text) => {
      const calls = parseToolCalls(text.replace(/<TOOL_RESPONSE>[\s\S]*?<\/TOOL_RESPONSE>/g, ''));
      for (; handledCalls < calls.length; handledCalls++) {
        const call = calls[handledCalls];
        console.log(`tool call ${call.name} ${JSON.stringify(call.arguments)}`);
        turnCalls++;
        const repeatRecall = call.name === 'recall' && turnRecalled;
        if (call.name === 'recall') turnRecalled = true;
        toolQueue = toolQueue.then(async () => {
          const started = Date.now();
          const out =
            turnCalls > MAX_CALLS_PER_TURN ? 'Too many tool calls. Stop calling tools and answer the user now.'
            : repeatRecall ? 'Already searched for this turn: answer from the result above.'
            : call.name === '(unparsable)' ? 'That tool call was not valid JSON. Answer the user without it.'
            : await runTool(call).catch((err) => `The tool failed: ${err}`);
          await model.toolOutput(out);
          console.log(`tool ${call.name} → "${out.slice(0, 160)}" (${Date.now() - started}ms incl. injecting)`);
        });
      }
    };

    // ---- turns: the model's transcripts are cumulative over the session; cut them into turns
    let userFrom = 0; // offsets into the cumulative texts
    let agentFrom = 0;
    let lastUserAt = 0;
    let lastAgentAt = 0;
    let latencyPending = false;
    model.onUserText = (text) => {
      if (!userText) [turnCalls, turnRecalled] = [0, false]; // a new user turn
      lastUserAt = Date.now();
      userText = text.slice(userFrom).trim();
      latencyPending = true;
      clearTimeout(prefetchTimer);
      prefetchTimer = setTimeout(() => void prefetchMemory(), PREFETCH_AFTER_MS);
    };

    // ---- When to search memory is decided here, not by the model: VoiceChat answers "where did
    // Rachel move?" from its own knowledge instead of calling recall. Once per user turn (when the model
    // starts replying, or after a pause), Laya routes what the model heard (~10 ms) while recall runs in
    // parallel (~60 ms); a memory turn gets the call + result forced in, as if the model had called it.
    // A turn is a memory turn if Laya says "past", it explicitly refers back ("you told me", "last
    // time"), or it is in the first person and recall finds a strong match. benchmarks/router
    // (combined.ts): 84%+ of memory questions, ~20% of other turns; an empty search is only injected
    // for first-person turns, so a false alarm on "tell me a joke" doesn't become "I don't remember".
    // While a turn is being routed the model may already be answering ("I am sorry, I do not…"):
    // its audio is held, then dropped if memory gets forced in, or played if not.
    let held: Int16Array[] | null = null;
    let holdTimer: NodeJS.Timeout | undefined;
    const holdAudio = () => {
      if (held) return;
      held = [];
      holdTimer = setTimeout(() => releaseAudio(true), MAX_HOLD_MS);
    };
    const releaseAudio = (playHeld: boolean) => {
      clearTimeout(holdTimer);
      const frames = held ?? [];
      held = null;
      if (playHeld) frames.forEach(play);
    };
    let prefetchTimer: NodeJS.Timeout | undefined;
    let prefetchedTurn = -1; // userFrom of the last turn routed
    const modelCalls = () => parseToolCalls(model.functionText.replace(/<TOOL_RESPONSE>[\s\S]*?<\/TOOL_RESPONSE>/g, '')).length;
    const callOpen = () => /<TOOLCALL>(?![\s\S]*<\/TOOLCALL>)/.test(model.functionText);
    async function prefetchMemory() {
      const heard = userText.replace(/^[\s,.?!;:-]+/, '');
      if (!heard || prefetchedTurn === userFrom) return;
      prefetchedTurn = userFrom;
      holdAudio();
      const callsBefore = modelCalls();
      const started = Date.now();
      const [route, found] = await Promise.all([
        layaRoute(heard).catch(() => 'chat' as const),
        recallBrief({ question: heard, excludeRoomSid: room.sid }).catch((err) => {
          console.error('recall failed', err);
          return { text: '', top: -Infinity, hits: 0 };
        }),
      ]);
      if (!found.text) return releaseAudio(true);
      const firstPerson = FIRST_PERSON.test(heard);
      const memoryTurn = route === 'past' || REFERS_BACK.test(heard) || (firstPerson && found.top >= STRONG_MATCH);
      console.log(`route "${heard}": laya ${route}, best match ${found.top.toFixed(1)} → ${memoryTurn ? 'memory' : 'no memory'} (${Date.now() - started}ms)`);
      // not a memory turn, or the model called a tool itself meanwhile (or is mid-call): its answer stands
      if (!memoryTurn || (!found.hits && !firstPerson && !REFERS_BACK.test(heard)) || modelCalls() > callsBefore || callOpen())
        return releaseAudio(true);
      handledCalls++; // the forced call shows up on the function channel: don't run it again
      turnCalls++;
      turnRecalled = true;
      answeringFromMemory = true;
      toolQueue = toolQueue.then(async () => {
        await model.toolOutput(found.text, { name: 'recall', arguments: { question: heard } });
        releaseAudio(false); // what it started saying before the result: dropped
        agentFrom = model.assistantText.length; // ...and left out of the transcript
        console.log(`prefetched recall → "${found.text.slice(0, 160)}" (${Date.now() - started}ms)`);
      });
    }
    const reply = () => model.assistantText.slice(agentFrom).trim();
    const turnTimer = setInterval(() => {
      const now = Date.now();
      const lastText = model.textEvents.at(-1);
      // first reply token after the user spoke: voice-to-voice latency
      if (latencyPending && lastText && lastText.at > lastUserAt) {
        latencyPending = false;
        void prefetchMemory(); // the model decided the user finished: route now rather than wait
        const durationMs = lastText.at - lastUserAt;
        console.log(`voice-to-voice latency: ${durationMs}ms`);
        report({ type: 'agentMetrics', data: { type: 'turn_latency', at: lastText.at, durationMs, participant: agentIdentity } });
      }
      if (lastText) lastAgentAt = lastText.at;
      if (userText && now - lastUserAt > TURN_QUIET_MS && (lastAgentAt > lastUserAt || now - lastUserAt > 3 * TURN_QUIET_MS)) {
        report({ type: 'transcript', data: { role: 'user', text: userText, participant: user, at: lastUserAt } });
        userFrom = model.userText.length;
        userText = '';
      }
      if (reply() && now - lastAgentAt > TURN_QUIET_MS && now - lastUserAt > TURN_QUIET_MS) {
        const text = reply().replace(/<[^>]+>/g, '').trim();
        if (text) report({ type: 'transcript', data: { role: 'assistant', text, participant: agentIdentity, at: lastAgentAt, fromMemory: answeringFromMemory || undefined } });
        agentFrom = model.assistantText.length;
        answeringFromMemory = false;
      }
    }, 200);

    // ---- audio out: the model's voice, every 80 ms (silence included: it is full duplex)
    const source = new AudioSource(OUTPUT_RATE, 1);
    await ctx.room.localParticipant!.publishTrack(
      LocalAudioTrack.createAudioTrack('voicechat', source),
      new TrackPublishOptions({ source: TrackSource.SOURCE_MICROPHONE }),
    );
    const play = (pcm: Int16Array) =>
      void source.captureFrame(new AudioFrame(pcm, OUTPUT_RATE, 1, pcm.length)).catch(() => undefined);
    model.onAudio = (pcm) => (held ? held.push(pcm) : play(pcm));

    // ---- audio in: the user's microphone at 16 kHz, in the model's 80 ms frames
    let listening = false;
    const listen = async (track: RemoteTrack, identity: string) => {
      if (listening || track.kind !== TrackKind.KIND_AUDIO) return;
      listening = true;
      user = identity;
      console.log(`listening to ${identity}`);
      let buffer = new Int16Array(0);
      for await (const frame of new AudioStream(track, { sampleRate: 16_000, numChannels: 1 })) {
        const merged = new Int16Array(buffer.length + frame.data.length);
        merged.set(buffer);
        merged.set(frame.data, buffer.length);
        let at = 0;
        for (; at + FRAME_SAMPLES <= merged.length; at += FRAME_SAMPLES) model.push(merged.slice(at, at + FRAME_SAMPLES));
        buffer = merged.slice(at);
      }
      listening = false;
    };
    ctx.room.on(RoomEvent.TrackSubscribed, (track, _pub, participant) => void listen(track, participant.identity));
    for (const p of ctx.room.remoteParticipants.values())
      for (const pub of p.trackPublications.values()) if (pub.track) void listen(pub.track as RemoteTrack, p.identity);

    ctx.addShutdownCallback(async () => {
      clearInterval(turnTimer);
      model.close();
    });
  },
});

/** Laya's 3-way route of one user turn (benchmarks/router): past / action / chat. */
async function layaRoute(text: string): Promise<'past' | 'action' | 'chat'> {
  const res = await fetch(`${env('LAYA_BASE_URL')}/v1/systemone`, {
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

function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set (see .env at the repo root)`);
  return value;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
}
