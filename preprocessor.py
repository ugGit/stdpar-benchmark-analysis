import pandas as pd
import re

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
  df.loc[(df['target_mode']=='gpu'), 'target'] = 'Nvidia A6000'
  geforce_benchmarks = df.query("benchmark.str.contains('geforce_2080')")['benchmark']
  geforce_mask = df['benchmark'].isin(geforce_benchmarks)
  df.loc[geforce_mask, 'target'] = 'Nvidia GeForce 2080'

  # Reorder columns in dataframe
  df = df[['benchmark', 'programming_model', 'target_mode', 'environment', 'target', 'algorithm', 'dataset', 'activations', 'kernel_time', 'cpu_time', 'time_unit', 'iterations', 'repetitions']]
  
  return df
