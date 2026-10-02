// Prefills today's prompt in the VoiceChat sidecar (~1 min) so the first call starts instantly.
import { env } from '@voice/config';
import { VoiceChatSession } from '@voice/voicechat';
import { prompt } from './prompt.js';

const started = Date.now();
(await VoiceChatSession.open(env('VOICECHAT_URL'), prompt())).close();
console.log(`voicechat prompt ready (${Date.now() - started}ms)`);
