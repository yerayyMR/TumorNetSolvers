import json

def save_results_to_json(results, output_file):
    """
    Saves the computed metrics to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump({"sample_metrics": results}, f, indent=4)
