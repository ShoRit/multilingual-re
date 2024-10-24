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

    # Loop through each entry in the train data, if it exists
    if 'train' in json_data:
        for entry in json_data["train"]:
            for dependency in entry["dep_graph"]:
                # Add the dependency type (the second element in the list) to the set
                dependency_types.add(dependency[1])

# Convert the set to a sorted list
unique_dependency_types = sorted(dependency_types)

# Print the unique dependency types
print(unique_dependency_types)

