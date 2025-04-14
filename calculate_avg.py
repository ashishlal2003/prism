import json


def calculate_average_time(file_path):
    try:
        with open(file_path, "r") as file:
            queries = json.load(file)

        total_time = 0
        count = 0

        for query in queries:
            total_time += query["OverAll Time : "]
            count += 1

        average_time = total_time / count if count > 0 else 0

        print(f"\nTraditional RAG average time is: {average_time:.10f}s")
        print("-" * 50)

    except FileNotFoundError:
        print("The specified file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
    except KeyError:
        print("The expected key was not found in the JSON data.")


def calculate_average_prism_times(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        total_times = {
            "Document Search Time": 0,
            "Section Search Time": 0,
            "Paragraph Search Time": 0,
            "Total Retrieval Time": 0,
        }

        count = len(data)

        for entry in data:
            total_times["Document Search Time"] += entry.get("Document Search Time", 0)
            total_times["Section Search Time"] += entry.get("Section Search Time", 0)
            total_times["Paragraph Search Time"] += entry.get(
                "Paragraph Search Time", 0
            )
            total_times["Total Retrieval Time"] += entry.get("Total Retrieval Time", 0)

        averages = {key: value / count for key, value in total_times.items()}

        print("\n=== PRISM Average Times ===")
        print("-" * 50)
        for key, value in averages.items():
            print(f"{key:<25}: {value:.10f}s")
        print("-" * 50)

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


file_path = "prism_data.json"
calculate_average_prism_times(file_path)


file_path = "rag_data.json"
calculate_average_time(file_path)
