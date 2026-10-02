import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dataPath, env } from '@voice/config'

const DIR = `${dataPath('benchmarks', 'voicechat', 'audio')}/`
const VOICE = 'am_michael'

export async function speak(text: string): Promise<Int16Array> {
  mkdirSync(DIR, { recursive: true })
  const file = `${DIR}${createHash('sha1').update(`${VOICE}:${text}`).digest('hex').slice(0, 16)}.pcm`
  if (existsSync(file)) {
    const b = readFileSync(file)
    return new Int16Array(b.buffer, b.byteOffset, b.byteLength / 2)
  }
  const res = await fetch(`${env('SPEECH_BASE_URL')}/audio/speech`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: env('TTS_MODEL'), voice: VOICE, input: text, response_format: 'wav' }),
  })
  if (!res.ok) throw new Error(`tts: ${res.status} ${await res.text()}`)
  const pcm = resample(parseWav(Buffer.from(await res.arrayBuffer())), 16_000)
  writeFileSync(file, Buffer.from(pcm.buffer))
  return pcm
}

function parseWav(b: Buffer): { samples: Float32Array; rate: number } {
  let off = 12
  let rate = 0,
    bits = 16,
    channels = 1,
    format = 1
  while (off < b.length) {
    const id = b.toString('ascii', off, off + 4)
    const size = b.readUInt32LE(off + 4)
    if (id === 'fmt ') {
      format = b.readUInt16LE(off + 8)
      channels = b.readUInt16LE(off + 10)
      rate = b.readUInt32LE(off + 12)
      bits = b.readUInt16LE(off + 22)
    } else if (id === 'data') {
      const data = b.subarray(off + 8, off + 8 + size)
      const n = data.length / (bits / 8) / channels
      const samples = new Float32Array(n)
      for (let i = 0; i < n; i++) {
        const at = i * channels * (bits / 8)
        samples[i] = format === 3 ? data.readFloatLE(at) : data.readInt16LE(at) / 32768
      }
      return { samples, rate }
    }
    off += 8 + size + (size % 2)
  }
  throw new Error('wav without data chunk')
}

function resample({ samples, rate }: { samples: Float32Array; rate: number }, to: number): Int16Array {
  const n = Math.floor((samples.length * to) / rate)
  const out = new Int16Array(n)
  for (let i = 0; i < n; i++) {
    const x = (i * rate) / to
    const j = Math.floor(x)
    const v = samples[j] + ((samples[Math.min(j + 1, samples.length - 1)] ?? 0) - samples[j]) * (x - j)
    out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)))
  }
  return out
}
