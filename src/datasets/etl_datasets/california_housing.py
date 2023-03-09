from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch

from src.common.constants import *
from src.datasets.experiment_dataset import ExperimentData, ExperimentDataset
from src.datasets.etl_utils import *


class CaliforniaHousing(ExperimentDataset):

    def get_data_without_target_embedding(self, params: Dict) -> ExperimentData:
        # Load data
        print('Loading CALIFORNIA HOUSING data...')
        data = fetch_california_housing(as_frame=True)
        df = data['frame']
        target_cols = ['MedHouseVal']
        cont_cols = [c for c in df.columns if c not in target_cols]
        cat_cols = []
        print(df.columns)

        print('Preprocessing continuous features...')
        targets_df = df[target_cols]
        df, _ = preprocess_continous_features(cont_feature_names=cont_cols,
                                              cat_features_name=cat_cols,
                                              n_quantiles=params[N_QUANTILE],
                                              output_distribution=params[OUTPUT_DISTRIBUTION],
                                              df_train=df.drop(target_cols, axis=1),
                                              df_test=None,
                                              discretize=False)
        df = pd.DataFrame(data=np.concatenate([df.values, targets_df.values], axis=1),
                          columns=list(df.columns) + target_cols)

        # Split train-test
        print('Splitting train/test data...')
        data_train, data_test = train_test_split(df, test_size=params[PERCENTAGE_TEST], random_state=0)

        # Features-target split
        X_train, y_train = data_train.drop(target_cols, axis=1), data_train[target_cols]
        X_test, y_test = data_test.drop(target_cols, axis=1), data_test[target_cols]

        # Data-->Torch
        print('Summary data:')
        torch_x_train = torch.tensor(X_train[list(X_train.columns)].values, dtype=torch.float)
        torch_y_train = torch.tensor(y_train[target_cols].values, dtype=torch.float)
        print('\tTrain Features: {}'.format(torch_x_train.shape))
        print('\tTrain Targets: {}'.format(torch_y_train.shape))
        torch_x_test = torch.tensor(X_test[list(X_train.columns)].values, dtype=torch.float)
        torch_y_test = torch.tensor(y_test[target_cols].values, dtype=torch.float)
        print('\tTest Features: {}'.format(torch_x_test.shape))
        print('\tTest Targets: {}'.format(torch_y_test.shape))

        data = ExperimentData(x_train=torch_x_train,
                              y_train=torch_y_train,
                              x_test=torch_x_test,
                              y_test=torch_y_test,
                              con_num=len(cont_cols),
                              cat_num=len(cat_cols),
                              cat_degrees=[],
                              problem_type=params[PROBLEM_TYPE],
                              problem_size=params[PROBLEM_SIZE],
                              weights=None,
                              aux=None)

        return data

    def get_data_with_target_embedding(self, params: Dict) -> ExperimentData:
        return None, None, None, None
