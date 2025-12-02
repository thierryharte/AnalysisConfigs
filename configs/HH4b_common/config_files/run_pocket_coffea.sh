#!/usr/bin/env bash

cleanup() {
    echo "Cleaning up..."
    [ -f "$new_config_template" ] && rm -f "$new_config_template"
    [ -f "$onnx_exec" ] && rm -f "$onnx_exec"
}

trap 'cleanup; kill -- "$pid"; exit 1' SIGINT SIGTERM

# Required args
config_options=${1%.py}
config_template=$2
run_options=$3
output=$4
shift 4

# All extra args initially
raw_extra_args=("$@")

# Detect and remove --test and --debug
is_test=false
is_debug=false
extra_args=()

for arg in "${raw_extra_args[@]}"; do
    case "$arg" in
        --test)
            is_test=true
            ;;
        --debug)
            is_debug=true
            ;;
        *)
            extra_args+=("$arg")
            ;;
    esac
done

# Script location
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Find free run number
i=0
while true; do
    candidate="${SCRIPT_DIR}/../__onnx_executor_${i}__.py"
    if [ ! -f "$candidate" ]; then
        echo "Found free Run number ${i}"
        onnx_exec="$candidate"
        break
    fi
    i=$((i+1))
done

cp "${SCRIPT_DIR}/../onnx_executor_common.py" "$onnx_exec"
new_config_template="__${i}_${config_template}"
cp "$config_template" "$new_config_template"

# Replace placeholder
sed -i "s/__config_file__/${config_options}/g" "$onnx_exec"
sed -i "s/__config_file__/${config_options}/g" "$new_config_template"

# Determine executor
hostname=$(hostname)
case "$hostname" in
    *t3*)
        EXECUTOR="dask@T3_CH_PSI"
        EXECUTOR_CUSTOM_SETUP="--executor-custom-setup ${onnx_exec}"
        ;;
    *lxplus*)
        EXECUTOR="dask@lxplus"
        EXECUTOR_CUSTOM_SETUP=""
        ;;
    *)
        echo "WARNING: Unknown hostname '$hostname', no executor set."
        EXECUTOR=""
        EXECUTOR_CUSTOM_SETUP=""
        ;;
esac

echo "Using executor: $EXECUTOR"

# Construct command
cmd=( pocket-coffea run
      --cfg "$new_config_template"
      --custom-run-options "$run_options"
      -o "$output"
      --process-separately
)

###################################
# No executor in test mode
###################################
if ! $is_test; then
    [[ -n "$EXECUTOR" ]] && cmd+=( -e "$EXECUTOR" )
    [[ -n "$EXECUTOR_CUSTOM_SETUP" ]] && cmd+=( $EXECUTOR_CUSTOM_SETUP )
fi

###################################
# ✔️ If test mode, add the flag
###################################
$is_test && cmd+=( --test )

# Append cleaned extra args
cmd+=( "${extra_args[@]}" )

# Print the full command exactly as executed
echo "${cmd[@]}"

###################################
# DEBUG MODE → run without trapping
###################################
if $is_debug; then
    "${cmd[@]}"
    status=$?
    cleanup
    exit $status
fi

###################################
# NORMAL MODE → run in background
###################################
"${cmd[@]}" &
pid=$!

wait $pid
status=$?

cleanup
exit $status
