// Prefill today's agent prompt in the VoiceChat sidecar (~1 min) so calls start instantly. Run by meet.sh.
import { prompt } from './s2s_prompt.js';
import { VoiceChatSession } from './voicechat.js';

const t = Date.now();
(await VoiceChatSession.open(process.env.VOICECHAT_URL ?? 'ws://localhost:8200', prompt())).close();
console.log(`voicechat prompt ready (${Date.now() - t}ms)`);
