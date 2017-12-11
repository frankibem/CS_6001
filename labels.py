import os
import json

# Create the output directory if it doesn't exist
output_dir = './output'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Read the labels, assign ids and save to json
with open('labels.csv', 'r') as in_file:
    labels = [line.strip() for line in in_file]

label_dict = {}

for i in range(len(labels)):
    label_dict[labels[i]] = i

with open(os.path.join(output_dir, 'label.json'), 'w') as jfile:
    json.dump(label_dict, jfile, sort_keys=True, indent=2)
