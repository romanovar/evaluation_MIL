import argparse
import yaml
import pandas as pd
import numpy as np
from numpy.core._exceptions import UFuncTypeError

from cnn.keras_utils import build_path_results
from stability.utils import calculate_aggregated_performance


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)
dataset_name = config['dataset_name']
res_path = config['results_path']
pooling_operator = config['pooling_operator']
class_name = config['class_name']

set_name1 ='val_set_CV0'
set_name2 = 'val_set_CV1'
set_name3 ='val_set_CV2'
set_name4 = 'val_set_CV3'
set_name5 ='val_set_CV4'

file_names = [set_name1, set_name2, set_name3, set_name4, set_name5]
parent_folder_predictions = 'CV'

performance_path = build_path_results(res_path, dataset_name, pooling_operator,
                                             script_suffix=parent_folder_predictions,
                                             result_suffix='performance')

evaluation_file_name = "evaluation_performance_"
column_names = ['accuracy', 'dice', 'AUC_'+class_name]


def combine_all_files(files, path, prefix):
    all_df = pd.DataFrame()
    for file_suffix in files:
        df = pd.read_csv(path+prefix+file_suffix+'.csv')
        all_df = all_df.append(df)
    return all_df


def get_performance_values(df, columns):
    performance_values= []
    for column in columns:
        performance_values.append(df[column])
    return performance_values


def convert_illegal_values_nan(values_lists, illegal_value="--"):
    for list_nr in range(len(values_lists)):
        masked_array = []
        try:
            masked_array = np.ma.masked_equal(values_lists[list_nr].values, illegal_value)
            masked_array= masked_array.filled(np.nan)
        except UFuncTypeError:
            print("No illegal value found")
        if len(masked_array) > 0:
            values_lists[list_nr] = pd.Series(masked_array, dtype=float)
    return values_lists


def create_save_aggregation_file(files, path, prefix, columns):
    all_df = combine_all_files(files, path, prefix)
    performance_values = get_performance_values(all_df, columns)

    performance_values = convert_illegal_values_nan(performance_values, illegal_value="--")

    row_mean_values = calculate_aggregated_performance(all_df.columns.values, 'mean', performance_values)
    row_stddev_values = calculate_aggregated_performance(all_df.columns.values, 'stand dev', performance_values)

    all_df = all_df.append(row_mean_values)
    all_df = all_df.append(row_stddev_values)
    all_df.to_csv(path + 'mean_' + prefix + '.csv')


create_save_aggregation_file(file_names, performance_path, evaluation_file_name, column_names)
