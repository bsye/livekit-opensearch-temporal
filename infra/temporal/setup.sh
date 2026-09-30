#!/bin/sh
# Idempotent Temporal bootstrap, run by the temporal-admin-tools image.
#   schema    create/upgrade the Postgres databases (before the server starts)
#   namespace create the namespace and search attributes (after the server is up)
set -eu

SCHEMA_DIR=/etc/temporal/schema/postgresql/v12

wait_for() {
  until nc -z "$1" "$2"; do echo "waiting for $1:$2"; sleep 1; done
}

schema() {
  SQL="temporal-sql-tool --plugin postgres12 --ep ${POSTGRES_SEEDS} -p 5432 -u ${POSTGRES_USER} --pw ${POSTGRES_PWD}"
  wait_for "${POSTGRES_SEEDS}" 5432
  for db in temporal temporal_visibility; do
    [ "$db" = temporal ] && dir=temporal || dir=visibility
    $SQL --db "$db" create-database 2>/dev/null || true # fails if it already exists
    $SQL --db "$db" setup-schema -v 0.0 2>/dev/null || true # fails if already set up
    $SQL --db "$db" update-schema -d "$SCHEMA_DIR/$dir/versioned"
  done
}

namespace() {
  until temporal operator cluster health >/dev/null 2>&1; do echo "waiting for temporal"; sleep 1; done
  temporal operator namespace describe --namespace "${TEMPORAL_NAMESPACE}" >/dev/null 2>&1 \
    || temporal operator namespace create --namespace "${TEMPORAL_NAMESPACE}" --retention 720h
  # a new namespace takes a few seconds to reach the server's namespace cache
  until temporal operator search-attribute list --namespace "${TEMPORAL_NAMESPACE}" >/dev/null 2>&1; do
    echo "waiting for namespace ${TEMPORAL_NAMESPACE}"; sleep 2
  done
  # used by the RoomSession workflow (apps/livekit-temporal)
  for attr in RoomName:Keyword ParticipantIdentities:KeywordList; do
    temporal operator search-attribute create --namespace "${TEMPORAL_NAMESPACE}" \
      --name "${attr%%:*}" --type "${attr#*:}" 2>/dev/null || true # fails if it already exists
  done
  temporal operator search-attribute list --namespace "${TEMPORAL_NAMESPACE}" | grep -E 'RoomName|ParticipantIdentities'
}

"$@"
