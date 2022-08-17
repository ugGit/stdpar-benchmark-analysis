import pandas as pd
import numpy as np
import re

# Extract the number of threads based on benchmark name, assuming patterns follow the style ..._tXY_[no_]overclock...
def extract_number_of_threads(benchmark_name):
    begin_thread_num_idx = benchmark_name.find('_t')+2
    end_thread_num_idx = benchmark_name.find('_no_overclock') if benchmark_name.find('_no_overclock') > 0 else benchmark_name.find('_overclocked')
    return (int)(benchmark_name[begin_thread_num_idx:end_thread_num_idx])

# Extract the size of the partitions used based on algorithm name, assuming patterns follow the style ..._partition_XY.
def extract_partition_size(algorithm_name):
    prefix = '_partition_'
    partition_size_offset = algorithm_name.find(prefix) + len(prefix)
    return algorithm_name[partition_size_offset:]

# Perform the whole preprocessing
def transform_dataframe(df_input):
  # Work on a copy of the input df
  df = df_input.copy()

  # Rename columns for clarity
  df = df.rename(columns = {'name':'algorithm', 'real_time':'kernel_time', 'label':'dataset', 'filename':'benchmark'})

  # Remove unused columns
  df = df.drop(['run_name', 'run_type', 'family_index', 'per_family_instance_index', 'repetition_index', 'threads'], axis=1)

  # Update the algorithm column by extracting relevant info from the description
  df['algorithm'] = df['algorithm'].map(lambda row : re.split('/', row)[1])

  # Update the benchmark column to remove the json
  file_ending_offset = -len(".json")
  df['benchmark'] = df['benchmark'].map(lambda row : row[:file_ending_offset])

  # Split information for environment from benchmark to individual column
  benchmarks_in_isolated_environment = df.query("benchmark.str.contains('isolated_env')")['benchmark'].unique()
  isolated_env_mask = df['benchmark'].isin(benchmarks_in_isolated_environment)
  # Init all values with default
  df['environment'] = 'traccc'
  # Override selection
  df.loc[isolated_env_mask, 'environment'] = 'isolated'
  # Remove the trailling caracters from isolated benchmark names
  prefix_offset = len('stdpar_')
  suffix_offset = -len('_isolated_env')
  df.loc[isolated_env_mask, 'benchmark'] = df.loc[isolated_env_mask, 'benchmark'].map(lambda row : row[prefix_offset:suffix_offset])

  # Split the execution mode from the benchmark (multicore/gpu/single-core)
  cuda_benchmarks = df.query("benchmark.str.contains('cuda')")['benchmark']
  cuda_mask = df['benchmark'].isin(cuda_benchmarks)
  multicore_benchmarks = df.query("benchmark.str.contains('multicore')")['benchmark']
  multicore_mask = df['benchmark'].isin(multicore_benchmarks)
  gpu_benchmarks = df.query("benchmark.str.contains('gpu')")['benchmark']
  gpu_mask = df['benchmark'].isin(gpu_benchmarks)
  # Init all values with default
  df['target_mode'] = 'single-core'
  # Override selection
  df.loc[cuda_mask, 'target_mode'] = 'gpu'
  df.loc[gpu_mask, 'target_mode'] = 'gpu'
  df.loc[multicore_mask, 'target_mode'] = 'multicore'

  # Add column for technology (cuda/c++/stdpar)
  cpp_benchmarks = df.query("benchmark.str.contains('seq_cca')")['benchmark']
  cpp_mask = df['benchmark'].isin(cpp_benchmarks)
  # Init all values with default
  df['programming_model'] = 'stdpar'
  # Override selection
  df.loc[cuda_mask, 'programming_model'] = 'cuda'
  df.loc[cpp_mask, 'programming_model'] = 'cpp'

  # Strip the algorithm values to the relevant part
  df['algorithm'] = df['algorithm'].map(lambda a : re.split('stdpar_', re.split('cca_', a)[-1])[-1])
  df.loc[(df['algorithm']=='seq_cca'),'algorithm'] = 'sparse_ccl'

  # Extract and augment the dataframe with the partition size used for SV algorithms
  param_tuning_benchmarks = df.query("algorithm.str.contains('_partition_')")['algorithm']
  param_tuning_mask = df['algorithm'].isin(param_tuning_benchmarks)
  # Set default for all algorithms 
  df['partition_size'] = 1024 
  # Reset all sparse ccl invocations which are parallelized on detector level module
  df.loc[(df['algorithm'] == 'sparse_ccl'), 'partition_size'] = np.NaN 
  # Extract the partition size for parameter tuning related benchmarks
  df.loc[param_tuning_mask, 'partition_size'] = df.loc[param_tuning_mask, 'algorithm'].map(extract_partition_size) 
  # Set suitable format
  df['partition_size'] = pd.to_numeric(df['partition_size']).astype('Int16')
  # Remove extra from algorithms column
  df.loc[param_tuning_mask, 'algorithm'] = df.loc[param_tuning_mask, 'algorithm'].map(lambda x : x[:x.find('_partition')])

  # Update the dataset column to only include the mu value
  mu_value_offset = len('tt_bar')
  df['dataset'] = df['dataset'].map(lambda row : re.split('/', row)[-2][mu_value_offset:])

  # Augment dataset with number of activation within a dataset (spread across 10 events)
  total_activations_in_dataset = [
      ('mu20', 401913),
      ('mu40', 687499),
      ('mu60', 1023445),
      ('mu100', 1716732),
      ('mu200', 3239265),
      ('mu300', 4815527),
  ]
  df_tait = pd.DataFrame.from_records(total_activations_in_dataset, columns=['dataset', 'activations'])
  df = df.join(df_tait.set_index('dataset'), on='dataset')

  # Normalize data for single event in dataset
  num_events = 10
  df['activations'] = df['activations'] / num_events
  df['kernel_time'] = df['kernel_time'] / num_events
  df['cpu_time'] = df['cpu_time'] / num_events

  # Extract and augment processor names
  df['target'] = 'Intel Xeon Gold 5220'
  df.loc[(df['target_mode']=='gpu'), 'target'] = 'NVIDIA A6000'
  geforce_benchmarks = df.query("benchmark.str.contains('geforce_2080')")['benchmark']
  geforce_mask = df['benchmark'].isin(geforce_benchmarks)
  df.loc[geforce_mask, 'target'] = 'NVIDIA GeForce 2080'

  # Extract the number of CPU cores used
  overclock_benchmarks = df.query("benchmark.str.contains('overclock')")['benchmark']
  overclock_mask = df['benchmark'].isin(overclock_benchmarks)
  df['cpu_cores'] = 1
  df.loc[multicore_mask, 'cpu_cores'] = 72
  df.loc[overclock_mask, 'cpu_cores'] = df.loc[overclock_mask, 'benchmark'].map(extract_number_of_threads)

  # Add Core Efficiency score
  df['computing_units'] = df['cpu_cores']
  df.loc[(gpu_mask & ~geforce_mask), 'computing_units'] = 10752 # A6000
  df.loc[(gpu_mask & geforce_mask), 'computing_units'] = 3072 # GeForce2080
  df['omega'] = (df['kernel_time'] / df['activations']) * df['computing_units'] 
  df['omega_alt'] = (df['activations'] / df['computing_units']) / df['kernel_time'] # alternative calculation, different meaning

  # Reorder columns in dataframe
  df = df[['benchmark', 'programming_model', 'target_mode', 'cpu_cores', 'environment', 'target', 'algorithm', 'partition_size', 'omega', 'omega_alt', 'dataset', 'activations', 'kernel_time', 'cpu_time', 'time_unit', 'iterations', 'repetitions']]
  
  return df
