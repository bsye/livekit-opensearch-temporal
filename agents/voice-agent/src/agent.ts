// Local voice agent: Silero VAD → Whisper STT (speaches) → LLM (LM Studio) → Kokoro TTS (speaches).
// Every finalized conversation turn and pipeline metric is signalled to the room's
// Temporal RoomSession workflow (apps/livekit-temporal).
import { fileURLToPath } from 'node:url';
import {
  cli,
  defineAgent,
  metrics,
  ServerOptions,
  voice,
  type JobContext,
  type JobProcess,
} from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import * as silero from '@livekit/agents-plugin-silero';
import type { SignalDefinition } from '@temporalio/common';
import { connectTemporal, signalRoom } from 'livekit-temporal/client';
import { agentMetrics, transcript, type AgentMetric } from 'livekit-temporal/shared';

const INSTRUCTIONS = `You are a helpful voice assistant running fully on local models.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.`;

// Local servers don't check API keys, but the OpenAI client requires one
const LOCAL_API_KEY = 'local';

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
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
        useRealtime: false, // speaches serves the batch /audio/transcriptions API; VAD segments the audio
      }),
      llm: new openai.LLM({
        baseURL: env('LLM_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('LLM_MODEL'),
        reasoningEffort: 'none', // thinking adds seconds of latency per turn
      }),
      tts: new openai.TTS({
        baseURL: env('SPEECH_BASE_URL'),
        apiKey: LOCAL_API_KEY,
        model: env('TTS_MODEL'),
        voice: env('TTS_VOICE') as openai.TTSVoices, // Kokoro voice ids, not OpenAI's
      }),
    });

    await session.start({ agent: new voice.Agent({ instructions: INSTRUCTIONS }), room: ctx.room });
    await ctx.connect();

    const room = { sid: await ctx.room.getSid(), name: ctx.room.name };
    const agentIdentity = ctx.room.localParticipant?.identity ?? 'agent';
    const report = <T>(signal: SignalDefinition<[T]>, arg: T) =>
      signalRoom(temporal, room, signal, arg).catch((err) =>
        console.error(`temporal signal ${signal.name} failed`, err),
      );

    session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item, createdAt }) => {
      if (item.type !== 'message' || !item.textContent) return;
      report(transcript, {
        role: item.role,
        text: item.textContent,
        participant: item.role === 'assistant' ? agentIdentity : (userIdentity(ctx) ?? 'user'),
        at: createdAt,
        interrupted: item.interrupted || undefined,
      });
    });

    // The TTS StreamAdapter re-emits the wrapped TTS's metrics, so the same request arrives twice
    const reportedRequests = new Set<string>();
    session.on(voice.AgentSessionEventTypes.MetricsCollected, ({ metrics: m, createdAt }) => {
      if ('requestId' in m && m.requestId) {
        const key = `${m.type}:${m.requestId}`;
        if (reportedRequests.has(key)) return;
        reportedRequests.add(key);
      }
      metrics.logMetrics(m);
      const metric = toAgentMetric(m, createdAt);
      if (metric) report(agentMetrics, metric);
    });

    session.generateReply({ instructions: 'Greet the user in one short sentence.' });
  },
});

function toAgentMetric(m: metrics.AgentMetrics, at: number): AgentMetric | undefined {
  switch (m.type) {
    case 'stt_metrics':
      return { type: m.type, at, durationMs: m.durationMs, audioDurationMs: m.audioDurationMs };
    case 'eou_metrics':
      return {
        type: m.type,
        at,
        endOfUtteranceDelayMs: m.endOfUtteranceDelayMs,
        transcriptionDelayMs: m.transcriptionDelayMs,
      };
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
      return undefined; // VAD metrics fire continuously; not useful per turn
  }
}

/** The first remote participant; this agent serves one user per room. */
function userIdentity(ctx: JobContext): string | undefined {
  return ctx.room.remoteParticipants.values().next().value?.identity;
}

function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set (see .env at the repo root)`);
  return value;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
}
