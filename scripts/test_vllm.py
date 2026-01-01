import os

from openai import OpenAI

# Configuration
BASE_URL = "http://127.0.0.1:2242/v1"
API_KEY = "qjlhqdjlshilejnqe1131245dnjqdhfled"  # Replace with the key you used in --api-key

# Initialize the client (optional timeout via VLLM_TIMEOUT)
_timeout = os.getenv("VLLM_TIMEOUT")
if _timeout:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=float(_timeout),
    )
else:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

def get_model_name():
    """Fetches the first available model name from the vLLM server."""
    models = client.models.list()
    return models.data[0].id

def chat():
    model = get_model_name()
    print(f"Connected to vLLM. Using model: {model}")
    print("Type 'exit' or 'quit' to stop.\n")

    # Initialize message history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        try:
            full_response = vllm_stream_chat(messages, model)
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            print(f"\nError: {e}")


def vllm_stream_chat(messages, model):
    """Stream a response from vLLM using the same settings as this script."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body={"repetition_penalty": 1.1}  # vLLM specific tweaks if needed
    )

    print("Assistant: ", end="", flush=True)
    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content
    print("\n")
    return full_response

if __name__ == "__main__":
    chat()
