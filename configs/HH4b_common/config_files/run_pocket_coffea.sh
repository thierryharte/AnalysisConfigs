#!/usr/bin/env bash
cleanup() {
    echo "Cleaning up..."
    [ -f "$new_config_template" ] && rm -f "$new_config_template"
    [ -f "$onnx_exec" ] && rm -f "$onnx_exec"
}
# Usage: ./your_script.sh config_options config_template run_options output [--test]

trap 'cleanup; kill -- "$pid"; exit 1' SIGINT SIGTERM
config_options=${1%.py}
config_template=$2
run_options=$3
output=$4


# Find the directory this script is in
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Find out which runnumber can be used
found_free_runnumber=False
i=0
while [ "$found_free_runnumber" != "True" ]; do
    if [ ! -f "${SCRIPT_DIR}/../__onnx_executor_${i}__.py" ]; then
        echo "Found free Runnumber ${i}"
        onnx_exec="${SCRIPT_DIR}/../__onnx_executor_${i}__.py"
        found_free_runnumber=True
    else
        i=$((i + 1))
    fi
done

cp "${SCRIPT_DIR}/../onnx_executor_common.py" "$onnx_exec"
new_config_template="__${i}_${config_template}"
cp "$config_template" "$new_config_template"

# Replace placeholder
sed -i "s/__config_file__/${config_options}/g" "$onnx_exec"
sed -i "s/__config_file__/${config_options}/g" "$new_config_template"

# Determine the executor based on the hostname
hostname=$(hostname)
if [[ "$hostname" == *"t3"* ]]; then
    EXECUTOR="dask@T3_CH_PSI"
    EXECUTOR_CUSTOM_SETUP="--executor-custom-setup ${onnx_exec}"
elif [[ "$hostname" == *"lxplus"* ]]; then
    EXECUTOR="dask@lxplus"
    EXECUTOR_CUSTOM_SETUP=""
else
    echo "WARNING: Unknown hostname '$hostname', no executor set."
    EXECUTOR=""
    EXECUTOR_CUSTOM_SETUP=""
fi

echo "Using executor: $EXECUTOR"
# Run the command
if [[ " $@ " =~ "--test" ]]; then
    echo "pocket-coffea run --cfg ${new_config_template} --test --custom-run-options ${run_options} -o ${output} --process-separately"
	if [[ ! " $@ " =~ "--debug" ]]; then
		pocket-coffea run \
			--cfg "$new_config_template" \
			--test \
			--custom-run-options "$run_options" \
			-o "$output" \
			--process-separately &
			pid=$!
	fi
else
    echo "pocket-coffea run --cfg ${new_config_template} ${EXECUTOR:+-e $EXECUTOR} --custom-run-options ${run_options} -o ${output} ${EXECUTOR_CUSTOM_SETUP} --process-separately"
	if [[ ! " $@ " =~ "--debug" ]]; then
		pocket-coffea run \
			--cfg "$new_config_template" \
			${EXECUTOR:+-e $EXECUTOR} \
			--custom-run-options "$run_options" \
			-o "$output" \
			${EXECUTOR_CUSTOM_SETUP} \
			--process-separately &
			pid=$!
			echo $pid
	fi
fi
if [[ ! " $@ " =~ "--debug" ]]; then
	wait $pid
	status=$?

	cleanup
	exit $status
fi
