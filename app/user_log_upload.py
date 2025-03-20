import csv
import requests

# Configuration
USER_LOGS_FILE = "data/user_logs.csv"
CLASSIFY_ENDPOINT = "http://localhost:8000/classify"  # Update with your FastAPI server URL if different

def send_user_logs_to_classify():
    """Reads user logs from the CSV file and sends them to the classify endpoint."""
    try:
        with open(USER_LOGS_FILE, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row.get("Question utilisateur", "").strip()
                if not question:
                    continue  # Skip empty questions
                
                # Prepare the payload
                payload = {"question": question}
                
                # Send the request to the classify endpoint
                try:
                    response = requests.post(CLASSIFY_ENDPOINT, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    print(f"Question: {question}")
                    print(f"Classification Result: {result}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to classify question: {question}")
                    print(f"Error: {e}")
    except FileNotFoundError:
        print(f"User logs file not found: {USER_LOGS_FILE}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    send_user_logs_to_classify()