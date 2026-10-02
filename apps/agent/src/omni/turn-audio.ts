import type { AudioFrame } from '@livekit/rtc-node'

const AUDIO_WINDOW_MS = 60_000
const MAX_TURN_MS = 30_000
const SPEECH_ONSET_LEAD_MS = 500

export class TurnAudio {
  private frames: { at: number; pcm: Int16Array }[] = []
  private rate = 48_000
  private turnStart: number | undefined

  constructor(private readonly baseUrl: string) {}

  tap(audio: ReadableStream<AudioFrame> | AsyncIterable<AudioFrame>): ReadableStream<AudioFrame> {
    const source = (audio as AsyncIterable<AudioFrame>)[Symbol.asyncIterator]()
    return new ReadableStream<AudioFrame>({
      pull: async (controller) => {
        const { value, done } = await source.next()
        if (done) return controller.close()
        this.record(value)
        controller.enqueue(value)
      },
      cancel: async () => {
        await source.return?.()
      },
    })
  }

  speechStarted(at: number) {
    this.turnStart ??= at
  }

  async upload(): Promise<string | undefined> {
    const end = Date.now()
    const start = Math.max((this.turnStart ?? end - MAX_TURN_MS) - SPEECH_ONSET_LEAD_MS, end - MAX_TURN_MS)
    this.turnStart = undefined
    const frames = this.frames.filter((f) => f.at >= start)
    if (!frames.length) return undefined
    const pcm = new Int16Array(frames.reduce((n, f) => n + f.pcm.length, 0))
    let offset = 0
    for (const f of frames) {
      pcm.set(f.pcm, offset)
      offset += f.pcm.length
    }
    const res = await fetch(`${this.baseUrl}/audio/turns?rate=${this.rate}`, {
      method: 'POST',
      headers: { 'content-type': 'application/octet-stream' },
      body: Buffer.from(pcm.buffer),
    })
    if (!res.ok) throw new Error(`upload turn audio: ${res.status}`)
    return ((await res.json()) as { id: string }).id
  }

  private record(frame: AudioFrame) {
    this.rate = frame.sampleRate
    const pcm =
      frame.channels === 1
        ? frame.data.slice()
        : Int16Array.from({ length: frame.samplesPerChannel }, (_, i) => frame.data[i * frame.channels])
    const now = Date.now()
    this.frames.push({ at: now, pcm })
    while (this.frames.length && this.frames[0].at < now - AUDIO_WINDOW_MS) this.frames.shift()
  }
}
