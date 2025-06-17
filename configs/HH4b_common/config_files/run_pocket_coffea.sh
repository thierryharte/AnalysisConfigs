#!/usr/bin/env bash

# Usage: ./your_script.sh config_options config_template run_options output [--test]

config_options=$1
config_template=$2
run_options=$3
output=$4
testing=$5  # Optional flag: --test

# Find the directory this script is in
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Determine the executor based on the hostname
hostname=$(hostname)
if [[ "$hostname" == *"t3"* ]]; then
    EXECUTOR="dask@T3_CH_PSI"
    EXECUTOR_CUSTOM_SETUP="--executor-custom-setup ${SCRIPT_DIR}/../onnx_executor_common.py"
elif [[ "$hostname" == *"lxplus"* ]]; then
    EXECUTOR="dask@lxplus"
    EXECUTOR_CUSTOM_SETUP=""
else
    echo "WARNING: Unknown hostname '$hostname', no executor set."
    EXECUTOR=""
    EXECUTOR_CUSTOM_SETUP=""
fi

echo "Using executor: $EXECUTOR"

# Copy the config file
cp "${SCRIPT_DIR}/${config_options}.py" "${SCRIPT_DIR}/__config_file__.py"

# Run the command
if [[ "$testing" == "--test" ]]; then
    pocket-coffea run \
        --cfg "$config_template" \
        --test \
        --custom-run-options "$run_options" \
        -o "$output" \
        --process-separately
else
    pocket-coffea run \
        --cfg "$config_template" \
        ${EXECUTOR:+-e $EXECUTOR} \
        --custom-run-options "$run_options" \
        -o "$output" \
        ${EXECUTOR_CUSTOM_SETUP} \
        # --executor-custom-setup "$SCRIPT_DIR/../onnx_executor_common.py" \
        --process-separately
fi

