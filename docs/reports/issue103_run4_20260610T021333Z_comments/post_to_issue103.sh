#!/usr/bin/env bash
set -euo pipefail
repo="phead198708/BiGan"
issue="103"
comments_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/comments" && pwd)"
if [[ "true" == "true" ]]; then
  gh issue edit "${issue}" --repo "${repo}" --body-file "$(dirname "${BASH_SOURCE[0]}")/overview.md"
fi
for f in "${comments_dir}"/*.md; do
  echo "posting ${f}"
  gh issue comment "${issue}" --repo "${repo}" --body-file "${f}"
done
