import { fileURLToPath } from 'node:url'
import { NativeConnection, Worker } from '@temporalio/worker'
import { env } from '@voice/config'
import { TASK_QUEUE } from '@voice/temporal'
import * as activities from '@voice/temporal/activities'

const worker = await Worker.create({
  connection: await NativeConnection.connect({ address: env('TEMPORAL_ADDRESS') }),
  namespace: env('TEMPORAL_NAMESPACE'),
  taskQueue: TASK_QUEUE,
  workflowsPath: fileURLToPath(import.meta.resolve('@voice/temporal/workflows')),
  activities,
})
await worker.run()
