import type { Client } from '@temporalio/client';
import { signalRoom, type ActionDecision, type RoomRef, type RoomSignal } from '@voice/temporal';

/** Everything the agent observes goes to the room's workflow; a failed signal never breaks the call. */
export function roomReporter(temporal: Client, room: () => RoomRef) {
  const report = (signal: RoomSignal) => {
    signalRoom(temporal, room(), signal).catch((err) => console.error(`temporal ${signal.type} signal failed`, err));
  };
  const reportAction = (decision: ActionDecision) => report({ type: 'action', data: decision });
  return { report, reportAction };
}
