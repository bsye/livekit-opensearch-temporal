// Local voice agent, all on the Apple GPU:
// Silero VAD → Parakeet STT (mlx-audio) → LLM (LM Studio) → Kokoro TTS (mlx-audio).
// Every finalized conversation turn and pipeline metric is signalled to the room's
// Temporal RoomSession workflow (apps/livekit-temporal). Tools (tools.ts) are risk-tiered:
// reversible ones run immediately, consequential ones need the user's approval (approval.ts),
// and Laya audits every committed action afterwards (audit.ts).
import { fileURLToPath } from 'node:url';
import {
  cli,
  defineAgent,
  inference,
  llm,
  metrics,
  ServerOptions,
  voice,
  type JobContext,
  type JobProcess,
} from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import * as silero from '@livekit/agents-plugin-silero';
import { connectTemporal, signalRoom } from 'livekit-temporal/client';
import type { AgentMetric, GateDecision, RoomSignal } from 'livekit-temporal/shared';
import { ActionAuditor } from './audit.js';
import { createTools } from './tools.js';

const instructions = (now: Date) => `You are a helpful voice assistant running fully on local models.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.
Use tools only when the user asks for that action. After a tool runs, tell the user what it did.
If the user corrects a reminder or says undo, cancel it (and set the corrected one).
The current local time is ${now.toTimeString().slice(0, 5)}, ${now.toDateString()}.`;

// Local servers don't check API keys, but the OpenAI client requires one
const LOCAL_API_KEY = 'local';

// Turn-taking. VAD reports end of speech after VAD_SILENCE_MS of silence (the turn detector
// requires >= 250ms); LiveKit's audio turn detector (v1-mini, local CPU) then decides whether
// the user finished or is just pausing, waiting ENDPOINTING_MIN_DELAY_MS..MAX before committing.
// Silence alone can't tell the two apart: tuned short it split "There is a… agent orchestration"
// into two turns, tuned long it slowed every reply.
const VAD_SILENCE_MS = 300;
const ENDPOINTING_MIN_DELAY_MS = 300; // LiveKit's recommended values with the audio detector
const ENDPOINTING_MAX_DELAY_MS = 2500;
// Barge-in: only real speech interrupts the agent, not "okay"/"mm-hm" or a blip of echo.
// (The SDK's adaptive backchannel detector is a LiveKit Cloud model, so it isn't available.)
const INTERRUPTION_MIN_MS = 600;
const INTERRUPTION_MIN_WORDS = 2;

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load({ minSilenceDuration: VAD_SILENCE_MS });
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
        useRealtime: false, // batch /audio/transcriptions per VAD segment; parakeet takes ~60ms
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
      turnHandling: {
        // pinned: in dev mode the default is v1 on LiveKit Cloud, which a self-hosted server can't use
        turnDetection: new inference.TurnDetector({ version: 'v1-mini' }),
        endpointing: { minDelay: ENDPOINTING_MIN_DELAY_MS, maxDelay: ENDPOINTING_MAX_DELAY_MS },
        interruption: { mode: 'vad', minDuration: INTERRUPTION_MIN_MS, minWords: INTERRUPTION_MIN_WORDS },
        // start the LLM on the final transcript before the turn is confirmed; TTS waits for the
        // confirmed turn, otherwise discarded drafts get spoken and replies sound repeated
        preemptiveGeneration: { enabled: true, preemptiveTts: false },
      },
    });

    // Set once connected; tools only run after that
    let room = { sid: '', name: '' };
    let agentIdentity = 'agent';
    const report = (signal: RoomSignal) =>
      signalRoom(temporal, room, signal).catch((err) => console.error(`temporal ${signal.type} signal failed`, err));

    // What the user said recently (the auditor compares actions against it)
    const userTurns: string[] = [];
    const recentUserText = () => userTurns.slice(-2).join(' ');
    // Remember who the user is: transcripts can arrive after they left, when the room no longer lists them
    let lastUser = 'user';
    const currentUser = () => (lastUser = userIdentity(ctx) ?? lastUser);
    const reportDecision = (decision: GateDecision) => report({ type: 'gate', data: decision });
    const tools = createTools({
      temporal,
      room: () => room,
      user: currentUser,
      agent: () => agentIdentity,
      auditor: new ActionAuditor(env('LAYA_BASE_URL'), () => agentIdentity, recentUserText, reportDecision),
      report: reportDecision,
    });

    await session.start({
      agent: new voice.Agent({ instructions: instructions(new Date()), tools }),
      room: ctx.room,
    });
    await ctx.connect();

    room = { sid: await ctx.room.getSid(), name: ctx.room.name ?? '' };
    agentIdentity = ctx.room.localParticipant?.identity ?? 'agent';

    session.on(voice.AgentSessionEventTypes.ConversationItemAdded, ({ item, createdAt }) => {
      if (item.type !== 'message' || !item.textContent) return;
      if (item.role === 'user') userTurns.push(item.textContent);
      report({
        type: 'transcript',
        data: {
          role: item.role,
          text: item.textContent,
          participant: item.role === 'assistant' ? agentIdentity : currentUser(),
          at: createdAt,
          interrupted: item.interrupted || undefined,
        },
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
      if (metric) report({ type: 'agentMetrics', data: { ...metric, participant: agentIdentity } });
    });

    // Voice-to-voice latency: user stops speaking → agent starts speaking. VAD only reports
    // the end of speech after VAD_SILENCE_MS of silence, so that is added back in.
    let userStoppedAt: number | undefined;
    session.on(voice.AgentSessionEventTypes.UserStateChanged, ({ oldState, newState, createdAt }) => {
      if (oldState === 'speaking' && newState === 'listening') userStoppedAt = createdAt;
    });
    session.on(voice.AgentSessionEventTypes.AgentStateChanged, ({ newState, createdAt }) => {
      if (newState !== 'speaking' || userStoppedAt === undefined) return;
      const durationMs = createdAt - userStoppedAt + VAD_SILENCE_MS;
      userStoppedAt = undefined;
      console.log(`voice-to-voice latency: ${durationMs}ms`);
      report({
        type: 'agentMetrics',
        data: { type: 'turn_latency', at: createdAt, durationMs, participant: agentIdentity },
      });
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
