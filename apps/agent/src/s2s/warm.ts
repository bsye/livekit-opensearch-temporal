import { env } from '@voice/config'
import { VoiceChatSession } from '@voice/voicechat'
import { prompt } from './prompt.js'

const started = Date.now()
;(await VoiceChatSession.open(env('VOICECHAT_URL'), prompt())).close()
console.log(`voicechat prompt ready (${Date.now() - started}ms)`)
