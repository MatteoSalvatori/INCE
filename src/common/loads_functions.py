from typing import Dict, List
import json


def load_config_json(jsons: List[str]) -> List[Dict]:
    """Load json files

    :param jsons: List[str], list of json paths to load
    :return: List[Dict] with the info contained in the jsons
    """
    to_return = []
    for path in jsons:
        with open(path) as f:
            to_return.append(json.load(f))
    return to_return
