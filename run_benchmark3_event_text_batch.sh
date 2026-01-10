#!/bin/bash

# Configuration
NUM_RUNS=5  # Change this to run multiple times

# Benchmarks to test
BENCHMARKS=(
#    "outcome_text_partslist"
#    "effect_text_partslist"
#    "cause_text_partslist"
    "outcome_text_partsdescriptions"
    "effect_text_partsdescriptions"
    "cause_text_partsdescriptions"
)

# Models to test
MODELS=(
    "qwen/qwen3-vl-235b-a22b-instruct"
    "bytedance/ui-tars-1.5-7b"
    "gpt-5-mini"
    "gemini-2.5-flash"
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

# Track progress
TOTAL_COMBINATIONS=$((${#BENCHMARKS[@]} * ${#MODELS[@]} * NUM_RUNS))
CURRENT=0

echo "======================================"
echo "Benchmark3 Event Text Batch Runner"
echo "======================================"
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Models: ${#MODELS[@]}"
echo "Runs per combination: $NUM_RUNS"
echo "Total runs: $TOTAL_COMBINATIONS"
echo "======================================"
echo ""

# Loop through each combination
for benchmark in "${BENCHMARKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for run in $(seq 1 $NUM_RUNS); do
            CURRENT=$((CURRENT + 1))
            echo "[$CURRENT/$TOTAL_COMBINATIONS] Running: $benchmark | $model | Run $run/$NUM_RUNS"
            echo "Command: python ./main_benchmark3_event_text.py --benchmark $benchmark --model $model"
            
            python ./main_benchmark3_event_text.py --benchmark "$benchmark" --model "$model"
            
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                echo "❌ Error: Command failed with exit code $EXIT_CODE"
                echo "Continuing with next benchmark..."
            else
                echo "✅ Completed successfully"
            fi
            echo ""
        done
    done
done

echo "======================================"
echo "✅ All benchmark runs completed!"
echo "======================================"
