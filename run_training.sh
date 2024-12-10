#!/bin/bash

poetry shell

# set python path for duration of script
export PYTHONPATH=".:./extern/mast3r:./extern/mast3r/dust3r:./extern/mast3r/dust3r/croco:./extern/scannetpp"
FILE_NAME=experiments.mast3r_3d.train

# run script
# python -m $FILE_NAME $1
python -m $FILE_NAME start

while true; do
    python -m $FILE_NAME resume
    EXIT_CODE=$?
    echo "Script exited with exit code $EXIT_CODE."

    # Check if exit was due to a KeyboardInterrupt (exit code 130)
    if [ $EXIT_CODE -eq 130 ]; then
        echo "Script stopped by user. Exiting."
        break
    fi

    echo "Script crashed. Restarting in 5 seconds..."
    sleep 5
done