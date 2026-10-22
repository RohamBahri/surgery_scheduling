#!/usr/bin/env bash
# One-time VF branch cleanup, preserving both frozen VF and divergent w2b history.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# Untracked files are permitted: they may contain valuable local research.
# Git itself will refuse a checkout if one of them conflicts with a tracked file.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: Commit or stash changes to tracked files before pruning." >&2
  exit 2
fi
if [[ "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: bash scripts/retire_legacy_branches.sh [--dry-run]" >&2
  exit 2
fi
dry=0
[[ "${1:-}" == "--dry-run" ]] && dry=1
git fetch --prune origin
if ! git switch main; then
  echo "ERROR: Checkout blocked, possibly by an untracked filename collision." >&2
  echo "Move the specific conflicting untracked files aside, then rerun." >&2
  echo "For a .DS_Store collision, 'rm -f .DS_Store' is safe Finder-metadata cleanup." >&2
  exit 2
fi
git pull --ff-only origin main
vf_sha="36ba5c62e561527c0bb67547dd7d5e72e9a86372"
w2b_sha="05c2fbee93c4042436489912425c99ae934c5a34"
git merge-base --is-ancestor "$vf_sha" origin/main || {
  echo "ERROR: main does not contain frozen VF scientific source." >&2
  exit 2
}
preserve_tag() {
  local label="$1" sha="$2" old
  if git rev-parse -q --verify "refs/tags/$label" >/dev/null; then
    old="$(git rev-parse "refs/tags/$label^{commit}")"
    [[ "$old" == "$sha" ]] || {
      echo "ERROR: archival tag $label has unexpected SHA $old" >&2
      exit 2
    }
    echo "Already preserved: $label"
  elif (( dry )); then
    echo "Would create archival tag $label from $sha"
  else
    git tag -a "$label" "$sha" -m "Research source preserved during October 2026 VF cleanup"
    git push origin "refs/tags/$label"
    echo "Created archival tag $label"
  fi
}
preserve_tag "research/vf-followups-frozen-2026-10-22" "$vf_sha"
preserve_tag "archive/w2b-weekly-surrogates-2026-10-22" "$w2b_sha"
retired=(
 anchored-exact-adaptive
 anchored-regret-learning
 anchored-research-v3
 behavioral-advice-comparison
 daily-experiment
 decision-model-pilot-v1
 full-2013-audited
 paper-selection-12h
 theory-full-experiment
 theory-full-experiment-v3
 theory-single-split-budgeted
 w2b-weekly-surrogates
 weekly-supported-days
)
for name in "${retired[@]}"; do
  remote_sha="$(git ls-remote --heads origin "refs/heads/$name" | cut -f1)"
  if [[ -z "$remote_sha" ]]; then
    echo "Already deleted: $name"
    continue
  fi
  if [[ "$name" == "w2b-weekly-surrogates" ]]; then
    [[ "$remote_sha" == "$w2b_sha" ]] || {
      echo "ERROR: w2b branch moved since audit; refusing to delete it." >&2
      exit 2
    }
  else
    git merge-base --is-ancestor "$remote_sha" origin/main || {
      echo "ERROR: $name now contains unique commits; refusing deletion." >&2
      exit 2
    }
  fi
  if (( dry )); then
    echo "Would delete remote $name ($remote_sha)"
    continue
  fi
  git push origin --delete "$name"
  echo "Deleted remote branch $name"
  if git show-ref --verify --quiet "refs/heads/$name"; then
    local_sha="$(git rev-parse "refs/heads/$name")"
    if [[ "$local_sha" == "$remote_sha" ]]; then
      git branch -D "$name"
      echo "Deleted matching local branch $name"
    else
      echo "Kept local $name: its commits differ from audited remote."
    fi
  fi
done
if (( ! dry )); then
  git fetch --prune origin
  echo "Remaining remote branches:"
  git branch -r
fi
