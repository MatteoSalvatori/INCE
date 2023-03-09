from src.common.constants import *
from src.common.loads_functions import *
from src.datasets.dataset_factory import DatasetFactory
from src.trainers.standard_problem_to_opt import StandardProblemToOpt


def main():
    #parser = get_parser_general_test_without_optimization()
    #args = parser.parse_args()
    #data_json_path = args.dataset_config
    #brain_trainer_json_path = args.model_config
    data_json_path = "./src/datasets/json_config/california_housing.json"
    brain_trainer_json_path = "./src/models/json_config/no_te__in__cls_decoder/no_te__in__cls_decoder.json"
    dataset_params, brain_params = load_config_json([data_json_path, brain_trainer_json_path])
    dataset_data = DatasetFactory(dataset_params=dataset_params)()
    to_opt = StandardProblemToOpt(dataset_data=dataset_data,
                                  brain_params=brain_params,
                                  problem_type=dataset_params[PROBLEM_TYPE],
                                  problem_size=dataset_params[PROBLEM_SIZE],
                                  target_embedding=dataset_params[USE_TARGET_EMBEDDING])
    to_opt()
    print('============== END SCRIPT ==============')


if __name__ == "__main__":
    main()
