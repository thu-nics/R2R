#!/usr/bin/env bash
set -e

N_BACKGROUND=$1
TEST_IDS=(0 0 0 0 0 5 5 5 5 5 10 10 10 10 10 15 15 15 15 15 20 20 20 20 20 25 25 25 25 25)

pids=()

cleanup() {
    echo -e "\nCtrl+C detected. Killing background workers..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 130
}

trap cleanup SIGINT

echo "Starting $N_BACKGROUND background workers..."

for ((i=1; i<=N_BACKGROUND; i++)); do
    python test/test_speed.py \
        --input_file test/input_text.txt \
        --input_id "$i" \
        --is_background &
    pids+=("$!")
    sleep 1
done

echo "Background load ready."
sleep 2

# ====== foreground tests ======
total_speed=0
total_ratio=0
count=0

echo "Running foreground tests..."

for id in "${TEST_IDS[@]}"; do
    echo "Testing input_id=$id ..."

    out=$(python test/test_speed.py \
        --input_file test/input_text.txt \
        --input_id "$id" \
        --output_csv output/s1l1/${N_BACKGROUND}.csv)

    # parse two numbers: <speed> <llm_ratio>
    speed=$(echo "$out" | awk '{print $1}')
    ratio=$(echo "$out" | awk '{print $2}')

    echo "  speed = ${speed} tokens/s, llm_ratio = ${ratio}"

    total_speed=$(echo "$total_speed + $speed" | bc)
    total_ratio=$(echo "$total_ratio + $ratio" | bc)
    count=$((count + 1))
done

if [ "$count" -gt 0 ]; then
    avg_speed=$(echo "scale=3; $total_speed / $count" | bc)
    avg_ratio=$(echo "scale=4; $total_ratio / $count" | bc)

    echo "===================================="
    echo "Average speed: ${avg_speed} tokens/s"
    echo "Average llm_ratio: ${avg_ratio}"
    echo "===================================="
fi

cleanup
