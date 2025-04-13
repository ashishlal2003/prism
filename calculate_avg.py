import json

# Function to calculate average time from a JSON file
def calculate_average_time(file_path):
    try:
        # Load the JSON data from the specified file
        with open(file_path, 'r') as file:
            queries = json.load(file)

        # Initialize total time and count
        total_time = 0
        count = 0

        # Iterate through each query to sum the time values
        for query in queries:
            total_time += query["OverAll Time : "]
            count += 1

        # Calculate the average time
        average_time = total_time / count if count > 0 else 0

        print(f"Traditional RAG average time is: {average_time:.10f}s")
    
    except FileNotFoundError:
        print("The specified file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
    except KeyError:
        print("The expected key was not found in the JSON data.")


import json

def calculate_average_prism_times(file_path):
    try:
        # Load the JSON data from the specified file
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        # Initialize variables to store total times
        total_times = {
            'Document Search Time': 0,
            'Section Search Time': 0,
            'Paragraph Search Time': 0,
            'Search Time': 0,
            'Total Duration': 0
        }
        
        # Count number of entries
        count = len(data)
        
        # Sum up all time values
        for entry in data:
            total_times['Document Search Time'] += entry.get('Document Search Time', 0)
            total_times['Section Search Time'] += entry.get('Section Search Time', 0)
            total_times['Paragraph Search Time'] += entry.get('Paragraph Search Time', 0)
            total_times['Search Time'] += entry.get('Search Time', 0)
            total_times['Total Duration'] += entry.get('Total Duration', 0)
            
        # Calculate averages
        averages = {key: value/count for key, value in total_times.items()}
        
        # Print results in a formatted way
        print("\n=== Average Times ===")
        print("-" * 50)
        for key, value in averages.items():
            print(f"{key:<20}: {value:.10f}s")
        print("-" * 50)
        
        # Calculate overall average of all times
        all_times_avg = sum(averages.values()) / len(averages)
        print(f"\nPRISM Average Overall Time : {all_times_avg:.10f}s")
        
    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
file_path = 'prism_data.json'  # Replace with your actual file path
calculate_average_prism_times(file_path)


# Example usage
file_path = 'rag_data.json'  # Replace with your actual file path
calculate_average_time(file_path)


