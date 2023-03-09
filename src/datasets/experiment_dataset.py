from collections import namedtuple


ExperimentData = namedtuple('ExperimentData',
                            'x_train, y_train, x_test, y_test, con_num, cat_num, cat_degrees, problem_type, problem_size, weights, aux')


class ExperimentDataset(object):

    def get_data_without_target_embedding(self, params) -> ExperimentData:
        raise NotImplementedError

    def get_data_with_target_embedding(self, params) -> ExperimentData:
        raise NotImplementedError
