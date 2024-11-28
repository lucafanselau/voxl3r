#!/bin/bash

poetry shell
PYTHONPATH=".:extern/scannetpp" python -m models.surface_net_3d.train resume

while true; do
    PYTHONPATH=".:extern/scannetpp" python -m models.surface_net_3d.train resume
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