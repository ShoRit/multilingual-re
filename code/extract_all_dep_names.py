'''
This file is used to generate a list of all the dependency parses used in both datasets.
From the list, we use universaldependencies.org to find corresponding meanings.
'''

import json
import os
import glob

# Specify the root directory to search for JSON files
root_dir = '../data/'

# Initialize a set to store unique dependency types
dependency_types = set()

# Use glob to find all JSON files in all subfolders
for json_file in glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True):
    with open(json_file) as f:
        json_data = json.load(f)
    try:
        # Loop through each entry in the train data, if it exists
        if 'train' in json_data:
            for entry in json_data["train"]:
                for dependency in entry["dep_graph"]:
                    # Add the dependency type (the second element in the list) to the set
                    dependency_types.add(dependency[1])
        if 'validation' in json_data:
            for entry in json_data["train"]:
                for dependency in entry["dep_graph"]:
                    # Add the dependency type (the second element in the list) to the set
                    dependency_types.add(dependency[1])
        if 'test' in json_data:
            for entry in json_data["train"]:
                for dependency in entry["dep_graph"]:
                    # Add the dependency type (the second element in the list) to the set
                    dependency_types.add(dependency[1])
    except:
        pass
# Convert the set to a sorted list
unique_dependency_types = sorted(dependency_types)

# Print the unique dependency types
print(unique_dependency_types)

