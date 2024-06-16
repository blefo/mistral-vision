from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient

if __name__ == "__main__":
    api_key = "puueDsQhS9LXPdEnooegEai7NeT4oer4"  # This key has been revoked
    client = MistralClient(api_key=api_key)

    chat_response = client.chat(
        model=client.jobs.retrieve("be2e7a1c-21e3-458a-881c-9d9adca22cef").fine_tuned_model,
        messages=[ChatMessage(role='user', content='What is the best French cheese?')]
    )

    print(chat_response)