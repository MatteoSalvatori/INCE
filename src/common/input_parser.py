import argparse


def get_parser_general_test_without_optimization():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_config', type=str, required=True, help='Data set configuration file')
    parser.add_argument('-m', '--model_config', type=str, required=True, help='Model parameters file')
    return parser


def get_parser_general_cross_val_test_without_optimization():
    parser = get_parser_general_test_without_optimization()
    parser.add_argument('-o', '--optimization_config', type=str, required=True, help='Optimization configuration file')
    return parser


def get_parser_general_cross_val_test_with_optimization():
    parser = get_parser_general_cross_val_test_without_optimization()
    parser.add_argument('-r', '--ray_config', type=str, required=True, help='Model parameters file to optimize by Ray')
    return parser
