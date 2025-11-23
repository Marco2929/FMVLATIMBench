Setup
=====

1. `cp .env.template .env`
2. Add api key to .env
3. `uv sync`
4. `python main_grounding_dominik.py --input benchmark1_grounding/examples/object_recognition_single/OBJ_REC1`

Command collection
==================

- Run multiple benchmarks: `for i in {1..5}; do
    python main_grounding_dominik.py --input benchmark1_grounding/examples/object_recognition_single/OBJ_REC"$i" --saveresults --model qwen3 --benchmark localize
done`

Models
======

- qwen/qwen3-vl-8b-instruct
- qwen/qwen3-vl-30b-a3b-instruct
- qwen/qwen3-vl-235b-a22b-instruct
