#!/usr/bin/env bash
# Keeps the SQLite database and the LLM answer cache on a separate "data" branch,
# so the history the screen needs (z-score snapshots, run numbers, pick tracking)
# survives between GitHub Actions runs without bloating the main branch.
# Each save replaces the branch with a single commit, so it never grows.
#
#   scripts/data_branch.sh seed      # once, on your Mac: upload your current database
#   scripts/data_branch.sh restore   # download the latest database into this folder
#   scripts/data_branch.sh save      # upload this folder's database (used by the workflow)
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=data
DB=data_base_creation/market_data.db
CACHE=filtering_of_data/llm_cache.json

restore() {
  if ! git fetch --quiet --depth=1 origin "$BRANCH" 2>/dev/null; then
    echo "::error::There is no '$BRANCH' branch yet. Run 'scripts/data_branch.sh seed' on your Mac once to upload the database."
    exit 1
  fi
  for f in "$DB" "$CACHE"; do
    name=$(basename "$f")
    if git cat-file -e "FETCH_HEAD:$name" 2>/dev/null; then
      git cat-file blob "FETCH_HEAD:$name" > "$f.tmp" && mv "$f.tmp" "$f"
      echo "Restored $f ($(du -h "$f" | cut -f1))"
    else
      echo "No $name on the $BRANCH branch, starting without it"
    fi
  done
}

save() {
  if [ ! -s "$DB" ]; then
    echo "::error::$DB is missing, nothing to save"
    exit 1
  fi
  entries=$(printf '100644 blob %s\t%s\n' "$(git hash-object -w "$DB")" "$(basename "$DB")")
  if [ -s "$CACHE" ]; then
    entries+=$'\n'$(printf '100644 blob %s\t%s' "$(git hash-object -w "$CACHE")" "$(basename "$CACHE")")
  fi
  tree=$(printf '%s\n' "$entries" | git mktree)
  commit=$(git -c user.name="data-store" -c user.email="data-store@users.noreply.github.com" \
    commit-tree "$tree" -m "Data store $(date -u +%Y-%m-%dT%H:%MZ)")
  git push --quiet --force origin "$commit:refs/heads/$BRANCH"
  echo "Saved database to the $BRANCH branch"
}

case "${1:-}" in
  restore) restore ;;
  save|seed) save ;;
  *) echo "usage: $0 seed|restore|save"; exit 2 ;;
esac
