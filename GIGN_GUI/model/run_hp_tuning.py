import json
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, PROJECT_ROOT)

from GIGN_GUI.model.GIGN_hyperparameter_tuning import HyperParameter_tuning

if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    HyperParameter_tuning(**params)
