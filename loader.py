import json
from os import walk

# Extract benchmark runs from file and return list of dicts
def load_benchmarks_from_file(directory, filename):
  # Open the JSON file
  filepath = f'{directory}/{filename}'
  f = open(filepath)

  # Returns JSON object in form of a dictionary
  data = json.load(f)
  
  # Add a field to each entry referencing the filename
  for entry in data["benchmarks"]:
      entry["filename"] = filename
  
  # Return the benchmark entries in form of a list
  return data["benchmarks"]

# Load benchmarks from files in directory and return a list of dicts
def load_benchmarks_from_dir(directory):
  # Find list of files in directory (assumes there are only JSON files!)
  filenames = next(walk(directory), (None, None, []))[2]  # [] if no file
  filenames.sort()

  # Load each file and append result to final list
  benchmarks = []
  for filename in filenames:
      benchmarks = benchmarks + load_benchmarks_from_file(directory, filename)
      
  return benchmarks
