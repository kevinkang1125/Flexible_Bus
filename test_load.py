import json
def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


file_path = "./trajectories.json"  # Replace with your file path
data = load_json(file_path)
print(data[0]['actions'])