import json
import os


def append_to_json_file(new_data, file_path):
    """
    Appends a dictionary to a JSON file. If the file doesn't exist, it creates one.
    If the file contains a dictionary, it converts it into a list of dictionaries.
    If the file is empty or invalid, it starts with an empty list.

    :param file_path: Path to the JSON file.
    :param new_data: Dictionary to append to the JSON file.
    """

    # Check if the file exists
    if os.path.exists(file_path):
        # Open the file and load existing data
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
                # If the file contains a dictionary, convert it to a list
                if isinstance(data, dict):
                    data = [data]
                # If the file contains something else, raise an error
                elif not isinstance(data, list):
                    raise ValueError(
                        "JSON file must contain a list or a dictionary at the root."
                    )
            except json.JSONDecodeError:
                # If the file is empty or invalid, start with an empty list
                data = []
    else:
        # If the file doesn't exist, start with an empty list
        data = []

    # Append the new dictionary
    data.append(new_data)

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
