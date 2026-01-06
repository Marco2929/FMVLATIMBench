#!/usr/bin/env zsh

# Define the list of models
models=(
    "gemini-2.5-flash"
    "bytedance/ui-tars-1.5-7b"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "qwen/qwen3-vl-235b-a22b-instruct"
)

# Define the list of inputs
# Note: 'object_place/cage' is listed twice based on your original snippet
inputs=(
    "benchmark4_manipulation/examples/levels/no_command/no_command"
#    "benchmark4_manipulation/examples/levels/object_place/cage"
#    "benchmark4_manipulation/examples/levels/object_place/lamp"
#    "benchmark4_manipulation/examples/levels/object_extend/grass"
#    "benchmark4_manipulation/examples/levels/object_extend/pipe"
#    "benchmark4_manipulation/examples/levels/object_multi/balls"
#    "benchmark4_manipulation/examples/levels/object_multi/laser"
#    "benchmark4_manipulation/examples/levels/object_remove/barrier"
#    "benchmark4_manipulation/examples/levels/object_remove/super_ball"
#    "benchmark4_manipulation/examples/levels/object_rotate/laser"
#    "benchmark4_manipulation/examples/levels/object_rotate/pipe"
#    "benchmark4_manipulation/examples/levels/object_move/accelerator_tube"
#    "benchmark4_manipulation/examples/levels/object_move/caution_wall"
)

# Path to python executable
PYTHON_BIN="/home/mm/dev/git/FoundationModelsVLA/.venv/bin/python"
SCRIPT_PATH="main_benchmark4_manipulation_marco.py"

# Outer loop: Iterate through models
for model in "${models[@]}"; do
    echo "========================================"
    echo "Processing Model: $model"
    echo "========================================"

    # Middle loop: 5 runs per model
    for i in {1..1}; do
        echo "  Starting run $i of 5 for $model"

        # Inner loop: Iterate through inputs
        for input_path in "${inputs[@]}"; do
            sudo "$PYTHON_BIN" "$SCRIPT_PATH" --input "$input_path" --model "$model"
        done
    done
done