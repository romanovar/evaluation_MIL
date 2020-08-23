import argparse
import yaml
import pandas as pd
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

set_name1 ='test_set_CV1_0'
set_name2 = 'test_set_CV1_1'
set_name3 ='test_set_CV1_2'
set_name4 = 'test_set_CV1_3'
set_name5 ='test_set_CV1_4'

file_names = [set_name1, set_name2, set_name3, set_name4, set_name5]
parent_folder_predictions = 'subsets'

performance_path = build_path_results(res_path, dataset_name, pooling_operator,
                                             script_suffix=parent_folder_predictions,
                                             result_suffix='performance')

evaluation_file_name = "evaluation_performance_"
column_names = ['accuracy', 'dice', 'AUC_Cardiomegaly']


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


def create_save_aggregation_file(files, path, prefix, columns):
    all_df = combine_all_files(files, path, prefix)
    performance_values = get_performance_values(all_df, columns)

    row_mean_values = calculate_aggregated_performance(all_df.columns.values, 'mean', performance_values)
    row_stddev_values = calculate_aggregated_performance(all_df.columns.values, 'stand dev', performance_values)

    all_df = all_df.append(row_mean_values)
    all_df = all_df.append(row_stddev_values)
    all_df.to_csv(path + 'mean_' + prefix + '.csv')


create_save_aggregation_file(file_names, performance_path, evaluation_file_name, column_names)
