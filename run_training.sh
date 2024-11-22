#!/bin/bash

while true; do
    poetry shell
    PYTHONPATH=".:extern/scannetpp" python models/surface_net_3d/train.py
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
