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
  mask = df['benchmark'].isin(benchmarks_in_isolated_environment)
  df['environment'] = 'traccc'
  df.loc[mask, 'environment'] = 'isolated'
  
  # Remove the trailling caracters from isolated benchmark names
  prefix_offset = len('stdpar_')
  sufix_offset = -len('_isolated_env')
  df.loc[mask, 'benchmark'] = df.loc[mask, 'benchmark'].map(lambda row : row[prefix_offset:sufix_offset])

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

  # Reorder columns in dataframe
  df = df[['benchmark', 'environment', 'algorithm', 'dataset', 'activations', 'kernel_time', 'cpu_time', 'time_unit', 'iterations', 'repetitions']]

  return df
