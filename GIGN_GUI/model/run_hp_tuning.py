import json
import sys
import os

"""
This script runs hyperparameter tuning for GIGN using Ray Tune .
It is designed to be called from the GUI with parameters passed as a JSON string.
"""
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, PROJECT_ROOT)

from GIGN_GUI.model.hyperparameter_tuning import HyperParameter_tuning

if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    HyperParameter_tuning(**params)
