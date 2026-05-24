#!/bin/bash
# OPSIN round-trip eval runner.
#
# Requires a Java runtime for OPSIN (via py2opsin). If `java` is not already on
# your PATH, set JAVA_HOME and this script will add its bin/ directory.
#
# By default this runs eval/testset.json (a small public example). To run your
# own larger set, point IUPAC_NAMER_EVAL_DATA at a JSON file of the same shape:
#   {"compounds": [{"id": "...", "smiles": "..."}, ...]}
#
# Usage:
#   ./eval/run_eval.sh                 # full testset
#   ./eval/run_eval.sh --quick         # first 100 compounds
#   ./eval/run_eval.sh --smiles "CCO"  # single compound

set -e
if [ -n "$JAVA_HOME" ]; then
  export PATH="$JAVA_HOME/bin:$PATH"
fi
cd "$(dirname "$0")/.."
PYTHONPATH=. python eval/authoritative_eval.py "$@"
