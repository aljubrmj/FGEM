import json

def get_config_from_json(json_file):
    """Parse the configurations from the config json file provided.

    Args:
        json_file (str): file path to configuration json file

    Returns:
        dict: configuration in json format
    """
    
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict