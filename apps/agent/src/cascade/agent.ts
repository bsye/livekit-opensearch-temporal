import { fileURLToPath } from 'node:url';
import {
  cli,
  defineAgent,
  inference,
  metrics,
  ServerOptions,
  voice,
  type JobContext,
  type JobProcess,
} from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import * as silero from '@livekit/agents-plugin-silero';
import { env } from '@voice/config';
import { warmReranker } from '@voice/memory';
import { connectTemporal, type AgentMetric, type RoomRef } from '@voice/temporal';
import { roomReporter } from '../reporting.js';
import { createActions } from '../tools/actions.js';
import { ActionAuditor } from '../tools/audit.js';
import { createTools } from '../tools/index.js';
import { stripControlTokens } from './llm-filter.js';

const instructions = (now: Date) => `You are a helpful voice assistant running fully on local models.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.
Use tools only when the user asks for that action. After a tool runs, tell the user what it did.
Whenever the user asks about their own life, past, plans, purchases, people they know, or anything they
told you before, call recall first, every time, even if you searched earlier in this conversation.
Never say you have no information about the user's past without calling recall.
If the user corrects a reminder or says undo, cancel it (and set the corrected one).
The current local time is ${now.toTimeString().slice(0, 5)}, ${now.toDateString()}.`;

const LOCAL_API_KEY = 'local'; // the OpenAI client requires one; local servers ignore it

// Turn-taking. VAD reports end of speech after VAD_SILENCE_MS; the audio turn detector then decides
// from intonation whether the user is done or pausing. Silence alone either split sentences at
// natural pauses or, tuned longer, slowed every reply.
const VAD_SILENCE_MS = 300;
const ENDPOINTING_MIN_DELAY_MS = 300;
const ENDPOINTING_MAX_DELAY_MS = 2500;
// Only real speech interrupts the agent, not "okay" or a blip of echo.
const INTERRUPTION_MIN_MS = 600;
const INTERRUPTION_MIN_WORDS = 2;

/** STT → LLM → TTS, all on local models. */
export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load({ minSilenceDuration: VAD_SILENCE_MS });
    await warmReranker();
  },

  entry: async (ctx: JobContext) => {
    const temporal = await connectTemporal();

    const session = new voice.AgentSession({
      vad: ctx.proc.userData.vad as silero.VAD,
      stt: new openai.STT({
        baseURL: env('SPEECH_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('STT_MODEL'),
        language: 'en',
        useRealtime: false,
      }),
      llm: new openai.LLM({
        baseURL: env('LLM_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('LLM_MODEL'),
        reasoningEffort: 'none', // thinking adds seconds per turn
      }),
      tts: new openai.TTS({
        baseURL: env('SPEECH_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('TTS_MODEL'),
        voice: env('TTS_VOICE') as openai.TTSVoices,
      }),
      turnHandling: {
        // dev mode defaults to v1, which only runs on LiveKit Cloud
        turnDetection: new inference.TurnDetector({ version: 'v1-mini' }),
        endpointing: { minDelay: ENDPOINTING_MIN_DELAY_MS, maxDelay: ENDPOINTING_MAX_DELAY_MS },
        interruption: { mode: 'vad', minDuration: INTERRUPTION_MIN_MS, minWords: INTERRUPTION_MIN_WORDS },
        // the LLM starts on the final transcript, but TTS waits for the confirmed turn, or discarded drafts get spoken
        preemptiveGeneration: { enabled: true, preemptiveTts: false },
      },
    });

    let room: RoomRef = { sid: '', name: '' };
    let agentIdentity = 'agent';
    const { report, reportAction } = roomReporter(temporal, () => room);

    const userTurns: string[] = [];
    // transcripts can arrive after the user left, when the room no longer lists them
    let lastUser = 'user';
    const currentUser = () => (lastUser = ctx.room.remoteParticipants.values().next().value?.identity ?? lastUser);
    let answeringFromMemory = false;

    const auditor = new ActionAuditor(() => agentIdentity, () => userTurns.slice(-2).join(' '), reportAction);
    const actions = createActions({ temporal, room: () => room, user: currentUser, auditor });
    const tools = createTools(actions, {
      room: () => room,
      agent: () => agentIdentity,
      reportAction,
      onRecall: () => (answeringFromMemory = true),
    });

    await session.start({
      agent: voice.Agent.create({
        instructions: instructions(new Date()),
        tools,
        llmNode: async (agentCtx, chatCtx, toolCtx, settings) => {
          const stream = await voice.Agent.default.llmNode(agentCtx.agent, chatCtx, toolCtx, settings);
          return stream ? stripControlTokens(stream) : null;
        },
      }),
      room: ctx.room,
    });
    await ctx.connect();

    room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' };
    agentIdentity = ctx.room.localParticipant?.identity ?? 'agent';

    session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item, createdAt }) => {
      if (item.type !== 'message' || !item.textContent) return;
      if (item.role !== 'user' && item.role !== 'assistant') return;
      const role = item.role;
      if (role === 'user') userTurns.push(item.textContent);
      report({
        type: 'transcript',
        data: {
          role,
          text: item.textContent,
          participant: role === 'assistant' ? agentIdentity : currentUser(),
          at: createdAt,
          interrupted: item.interrupted || undefined,
          fromMemory: (role === 'assistant' && answeringFromMemory) || undefined,
        },
      });
      if (role === 'assistant') answeringFromMemory = false;
    });

    // the TTS StreamAdapter re-emits the wrapped TTS's metrics, so each request arrives twice
    const reportedRequests = new Set<string>();
    session.on(voice.AgentSessionEventTypes.MetricsCollected, ({ metrics: m, createdAt }) => {
      if ('requestId' in m && m.requestId) {
        const key = `${m.type}:${m.requestId}`;
        if (reportedRequests.has(key)) return;
        reportedRequests.add(key);
      }
      metrics.logMetrics(m);
      const metric = toAgentMetric(m, createdAt);
      if (metric) report({ type: 'agentMetrics', data: { ...metric, participant: agentIdentity } });
    });

    // Voice-to-voice: user stops speaking → agent starts. VAD reports the stop VAD_SILENCE_MS late.
    let userStoppedAt: number | undefined;
    session.on(voice.AgentSessionEventTypes.UserStateChanged, ({ oldState, newState, createdAt }) => {
      if (oldState === 'speaking' && newState === 'listening') userStoppedAt = createdAt;
    });
    session.on(voice.AgentSessionEventTypes.AgentStateChanged, ({ newState, createdAt }) => {
      if (newState !== 'speaking' || userStoppedAt === undefined) return;
      const durationMs = createdAt - userStoppedAt + VAD_SILENCE_MS;
      userStoppedAt = undefined;
      console.log(`voice-to-voice latency: ${durationMs}ms`);
      report({ type: 'agentMetrics', data: { type: 'turn_latency', at: createdAt, durationMs, participant: agentIdentity } });
    });

    session.generateReply({ instructions: 'Greet the user in one short sentence.' });
  },
});

function toAgentMetric(m: metrics.AgentMetrics, at: number): AgentMetric | undefined {
  switch (m.type) {
    case 'stt_metrics':
      return { type: m.type, at, durationMs: m.durationMs, audioDurationMs: m.audioDurationMs };
    case 'eou_metrics':
      return { type: m.type, at, endOfUtteranceDelayMs: m.endOfUtteranceDelayMs, transcriptionDelayMs: m.transcriptionDelayMs };
    case 'llm_metrics':
      return {
        type: m.type,
        at,
        durationMs: m.durationMs,
        ttftMs: m.ttftMs,
        promptTokens: m.promptTokens,
        completionTokens: m.completionTokens,
      };
    case 'tts_metrics':
      return { type: m.type, at, durationMs: m.durationMs, ttfbMs: m.ttfbMs, audioDurationMs: m.audioDurationMs };
    default:
      return undefined;
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
}
