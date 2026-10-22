#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

remove_volumes=false
for arg in "$@"; do
  case "$arg" in
    -v | --volumes)
      remove_volumes=true
      ;;
    -h | --help)
      echo "Usage: ./down.sh [-v|--volumes]"
      echo "  Stops dev-db containers and removes the compose network."
      echo "  -v, --volumes  Also delete the Postgres data volume (full reset)."
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try ./down.sh --help)" >&2
      exit 1
      ;;
  esac
done

if [[ "$remove_volumes" == true ]]; then
  echo "Stopping dev-db and removing volumes ..."
  docker compose down -v
else
  echo "Stopping dev-db (data volume kept; use ./down.sh -v to remove) ..."
  docker compose down
fi

echo "Done."
