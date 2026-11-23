#!/bin/bash
# run_arm.sh <output name> <map|plots> <extra run_agent_nav.py args...>
# One comparison run on a fixed task; interrupted 20 s after the agent reports done, then scored.
cd "$(dirname "$0")/.." || exit 1
name=$1; task=$2; shift 2
set -a; . ~/Desktop/code/rhizome-ai-agent/openrouter.env; set +a
export GEMINI_API_KEY=$(cat ~/Desktop/keys/robot/gemini.key) RHIZOME_ROOT=~/Desktop/code/rhizome-ai-agent
export RHIZOME_HOST_BUILD=$PWD/outputs/rhizome_host_skills/host RHIZOME_INFERENCE_PYTHON=~/Desktop/code/ml-aigen-tools/venv312/bin/python
unset ANTHROPIC_API_KEY
if [ "$task" = map ]; then
  task_args=(--scene outputs/staggered_soybean2 --spawn -6.0 -3.81 0 --seconds 2400
    --field '[[-6.3,-5.1],[3.8,-4.9],[4.0,0.6],[9.2,0.4],[9.0,5.2],[-1.0,4.9],[-1.1,-0.5],[-6.0,-0.3]]'
    --mission "$(.venv/bin/python -c "import json;print(json.load(open('outputs/shadow_map2/report.json'))['mission'])")")
else
  task_args=(--scene outputs/two_fields_wide5 --weed --spawn -16.5 1.524 0 --seconds 1800
    --mission "$(.venv/bin/python -c "import json;print(json.load(open('outputs/shadow_plots1/report.json'))['mission'])")")
fi
rm -rf outputs/$name
.venv/bin/python scripts/run_agent_nav.py --output outputs/$name --nav-device cuda --obstacle-max-height 0.25 \
  --nav-params '{"turn_pass_distance_m": 1.524}' "${task_args[@]}" "$@" > outputs/$name.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if grep -q '"kind": "agent_final"' outputs/$name/events.jsonl 2>/dev/null; then sleep 20; kill -INT $pid; break; fi
  sleep 10
done
wait $pid
.venv/bin/python scripts/score_run.py outputs/$name
