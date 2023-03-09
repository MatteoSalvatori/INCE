from typing import Dict

from src.common.constants import *
from src.datasets.etl_datasets.california_housing import CaliforniaHousing


class DatasetFactory(object):

    def __init__(self, dataset_params: Dict):
        self.dataset_params = dataset_params

    def __call__(self, *args, **kwargs):
        if self.dataset_params[DATASET_NAME] == 'California_Housing':
            return CaliforniaHousing().get_data_without_target_embedding(self.dataset_params)
