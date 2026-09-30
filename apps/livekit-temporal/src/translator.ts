// LiveKit webhook → Temporal translator.
// Verifies each webhook, then signal-with-starts the RoomSession workflow for that room.
// Responds 2xx only once Temporal has accepted the signal.
import http from 'node:http';
import { Client, Connection, WorkflowExecutionAlreadyStartedError } from '@temporalio/client';
import { WorkflowIdReusePolicy } from '@temporalio/common';
import { WebhookReceiver } from 'livekit-server-sdk';
import { livekitEvent, RoomName, TASK_QUEUE, type LiveKitEvent } from './shared.js';
import { roomSession } from './workflows.js';

const port = Number(process.env.TRANSLATOR_PORT ?? 3100);
const receiver = new WebhookReceiver(required('LIVEKIT_API_KEY'), required('LIVEKIT_API_SECRET'));
const client = new Client({
  connection: await Connection.connect({ address: process.env.TEMPORAL_ADDRESS ?? 'localhost:7233' }),
  namespace: process.env.TEMPORAL_NAMESPACE ?? 'default',
});

http
  .createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/healthz') return reply(res, 200, 'ok');
    if (req.method !== 'POST' || req.url !== '/webhook') return reply(res, 404, 'not found');

    let event: LiveKitEvent;
    try {
      const webhook = await receiver.receive(await readBody(req), req.headers.authorization);
      event = webhook.toJson() as unknown as LiveKitEvent;
    } catch (err) {
      console.warn('rejected webhook:', (err as Error).message);
      return reply(res, 401, 'invalid webhook');
    }

    // Workflow id = room sid: room names get reused, sids are unique per session
    const roomSid = event.room?.sid ?? event.egressInfo?.roomId;
    const roomName = event.room?.name ?? event.egressInfo?.roomName ?? event.ingressInfo?.roomName;
    if (!roomSid) {
      console.log(`${event.event} ${event.id}: no room sid, not forwarded`);
      return reply(res, 200, 'ignored');
    }

    try {
      await client.workflow.signalWithStart(roomSession, {
        workflowId: roomSid,
        taskQueue: TASK_QUEUE,
        signal: livekitEvent,
        signalArgs: [event],
        // a closed session must not be restarted by a late webhook
        workflowIdReusePolicy: WorkflowIdReusePolicy.REJECT_DUPLICATE,
        typedSearchAttributes: roomName ? [{ key: RoomName, value: roomName }] : [],
      });
      console.log(`${event.event} → ${roomSid}${event.participant ? ` (${event.participant.identity})` : ''}`);
      reply(res, 200, 'ok');
    } catch (err) {
      if (err instanceof WorkflowExecutionAlreadyStartedError) {
        console.warn(`${event.event} → ${roomSid}: session already closed, dropped`);
        return reply(res, 200, 'session closed');
      }
      console.error(`${event.event} → ${roomSid}: temporal error`, err);
      reply(res, 503, 'temporal unavailable');
    }
  })
  .listen(port, () => console.log(`translator listening on :${port}/webhook`));

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function reply(res: http.ServerResponse, status: number, body: string): void {
  res.writeHead(status, { 'content-type': 'text/plain' }).end(body);
}

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set (see .env at the repo root)`);
  return value;
}
