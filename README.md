# FmVlaTimBench

Benchmarking VLAs and VLMs again The Incredible Machine 2 (TIM2) with increasing difficulty.

## Setup

1. `cp .env.template .env`
2. Add OPENROUTER_API_KEY to .env and adjust BASE_URL if needed
3. `uv sync`
4. Make sure the virtual python env is activated: `uv venv` and check output

## Overview

Benchmarks 1-3 can be done with pure images only.
They are located in files main `main_benchmark1/2/3_*.py`. Run `--help` to see possible tasks.

Benchmarks 4-5 require mouse control and the game must be visible on your screen to take screenshots.

Add `--model qwen3/...` to change the model. Additional models are likely not to be implemented right now so you have to add it with a wrapper class and correct parsing and image preprocessing methods.

## Benchmark 1: Grounding

Model provider:
- qwen/qwen3-vl-235b-a22b-instruct: Parasail (fp8) and DeepInfra (fp8)
- qwen/qwen3-vl-8b-instruct: Parasail (bf16) and DeepInfra (fp8)
- bytedance/ui-tars-1.5-7b: Parasil (bf16)

`python main_benchmark1_grounding.py --benchmark qwen3_classify`

`python main_benchmark1_grounding.py --benchmark qwen3_localize`

`python main_benchmark1_grounding.py --benchmark qwen3_localize_multi`

`python main_benchmark1_grounding.py --benchmark uitars_localize`

## Benchmark 2: Understanding

`python main_benchmark2_understanding.py --benchmark with_instruct`

`python main_benchmark2_understanding.py --benchmark without_instruct`

`python main_benchmark2_understanding.py --benchmark state_indent`

## Benchmark 3: Event

`python main_benchmark3_event.py --benchmark outcome_text`

`python main_benchmark3_event.py --benchmark effect_text`

`python main_benchmark3_event.py --benchmark cause_text`

`python main_benchmark3_event_visual.py --benchmark outcome_visual`

`python main_benchmark3_event_visual.py --benchmark effect_visual`

`python main_benchmark3_event_visual.py --benchmark cause_visual`

## Scratchbook (to be deleted)

- qwen/qwen3-vl-8b-instruct
- qwen/qwen3-vl-30b-a3b-instruct
- qwen/qwen3-vl-235b-a22b-instruct
