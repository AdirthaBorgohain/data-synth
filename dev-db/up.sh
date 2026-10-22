#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Starting Postgres on localhost:5433 ..."
docker compose up -d db

echo "Waiting for database health ..."
docker compose up -d --wait db 2>/dev/null || true
until docker compose exec -T db pg_isready -U churn_admin -d churn_dev >/dev/null 2>&1; do
  sleep 1
done

echo "Seeding (wipe + reload) ..."
docker compose build seed
docker compose run --rm seed

echo ""
echo "Done. Row counts:"
docker compose exec -T db psql -U churn_admin -d churn_dev -c "
  SELECT relname AS table_name, n_live_tup AS approx_rows
  FROM pg_stat_user_tables
  WHERE schemaname IN ('public', 'analytics')
  ORDER BY schemaname, relname;
"

echo ""
echo "Set in project root .env:"
echo "  DATABASE_URL=postgresql+psycopg://datasynth_ro:datasynth_ro_dev@localhost:5433/churn_dev"
