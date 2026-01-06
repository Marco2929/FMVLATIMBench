#!/usr/bin/env zsh

#export model="gpt-5-mini"
#
#for i in {1..4}; do
#    echo "Starting run $i of 4"
#    python /home/mm/dev/git/FoundationModelsVLA/main_benchmark3_event_text.py --benchmark="cause_text" --model="$model"
#    python /home/mm/dev/git/FoundationModelsVLA/main_benchmark3_event_text.py --benchmark="effect_text" --model="$model"
#    python /home/mm/dev/git/FoundationModelsVLA/main_benchmark3_event_text.py --benchmark="outcome_text" --model="$model"
#done

export model="gpt-5-mini"

for i in {1..4}; do
    echo "Starting run $i of 4"
    python /home/mm/dev/git/FoundationModelsVLA/main_benchmark3_event_text.py --benchmark="cause_text" --model="$model"
    python /home/mm/dev/git/FoundationModelsVLA/main_benchmark3_event_text.py --benchmark="outcome_text" --model="$model"
done