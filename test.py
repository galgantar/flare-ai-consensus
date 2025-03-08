import requests
import json


def send_chat_request(system_message: str, user_message: str) -> dict | None:
    url = "http://localhost/api/routes/chat/"

    # Prepare the JSON payload
    payload = {"system_message": system_message, "user_message": user_message}

    # Set headers for JSON content
    headers = {"Content-Type": "application/json"}

    try:
        # Send POST request
        response = requests.post(url, json=payload, headers=headers)

        # Check if request was successful
        response.raise_for_status()

        # Return the response
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    system_message = "You are a helpful assistant"
    user_message = "Write me a a poem"

    result = send_chat_request(system_message, user_message)
    if result:
        print("Response:", json.dumps(result, indent=2))
