import http from 'node:http'
import { env } from '@voice/config'
import { connectTemporal, type LiveKitEvent, signalRoom } from '@voice/temporal'
import { WebhookReceiver } from 'livekit-server-sdk'

const receiver = new WebhookReceiver(env('LIVEKIT_API_KEY'), env('LIVEKIT_API_SECRET'))
const client = await connectTemporal()
const participantKinds = new Map<string, string>()

http
  .createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/healthz') return reply(res, 200, 'ok')
    if (req.method !== 'POST' || req.url !== '/webhook') return reply(res, 404, 'not found')

    let event: LiveKitEvent
    try {
      const webhook = await receiver.receive(await readBody(req), req.headers.authorization)
      event = webhook.toJson() as unknown as LiveKitEvent
    } catch (err) {
      console.warn('rejected webhook:', (err as Error).message)
      return reply(res, 401, 'invalid webhook')
    }

    if (event.participant) {
      const { sid, kind } = event.participant
      if (kind) participantKinds.set(sid, kind)
      else event.participant.kind = participantKinds.get(sid)
      if (event.event === 'participant_left') participantKinds.delete(sid)
    }

    const roomSid = event.room?.sid ?? event.egressInfo?.roomId
    const roomName = event.room?.name ?? event.egressInfo?.roomName ?? event.ingressInfo?.roomName
    if (!roomSid) {
      console.log(`${event.event} ${event.id}: no room sid, not forwarded`)
      return reply(res, 200, 'ignored')
    }

    try {
      if (!(await signalRoom(client, { sid: roomSid, name: roomName }, { type: 'livekitEvent', data: event }))) {
        console.warn(`${event.event} → ${roomSid}: session already closed, dropped`)
        return reply(res, 200, 'session closed')
      }
      console.log(`${event.event} → ${roomSid}${event.participant ? ` (${event.participant.identity})` : ''}`)
      reply(res, 200, 'ok')
    } catch (err) {
      console.error(`${event.event} → ${roomSid}: temporal error`, err)
      reply(res, 503, 'temporal unavailable')
    }
  })
  .listen(Number(env('TRANSLATOR_PORT')), () =>
    console.log(`translator listening on :${env('TRANSLATOR_PORT')}/webhook`),
  )

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (c: Buffer) => chunks.push(c))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function reply(res: http.ServerResponse, status: number, body: string): void {
  res.writeHead(status, { 'content-type': 'text/plain' }).end(body)
}
