import { fileURLToPath } from 'node:url';
import { NativeConnection, Worker } from '@temporalio/worker';
import * as activities from './activities.js';
import { TASK_QUEUE } from './shared.js';

const connection = await NativeConnection.connect({
  address: process.env.TEMPORAL_ADDRESS ?? 'localhost:7233',
});
const worker = await Worker.create({
  connection,
  namespace: process.env.TEMPORAL_NAMESPACE ?? 'default',
  taskQueue: TASK_QUEUE,
  workflowsPath: fileURLToPath(new URL('./workflows.ts', import.meta.url)),
  activities,
});
await worker.run();
