if __name__ == "__main__":
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.client import MistralClient

    api_key = "puueDsQhS9LXPdEnooegEai7NeT4oer4"  # This key has been revoked
    wandb_api_key = "062fb4bbe403b946cfaa4f52792fe7d7133c0274"  # This key has been revoked
    client = MistralClient(api_key=api_key)


    chat_response = client.chat(
        model=retrieved_job.fine_tuned_model,
        messages=[ChatMessage(role='user', content='What is the best French cheese?')]
    )