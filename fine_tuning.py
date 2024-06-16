import os
from mistralai.client import MistralClient
from mistralai.models.jobs import WandbIntegrationIn, TrainingParameters

api_key = "puueDsQhS9LXPdEnooegEai7NeT4oer4"    # This key has been revoked
wandb_api_key = "062fb4bbe403b946cfaa4f52792fe7d7133c0274"  # This key has been revoked
client = MistralClient(api_key=api_key)

if __name__ == "__main__":
    print('----- Starting file creation -----')
    with open("transformed_data/train.jsonl", "rb") as f:
        vision_mnist_train = client.files.create(file=("train.jsonl", f))

    with open("transformed_data/test.jsonl", "rb") as f:
        vision_mnist_test = client.files.create(file=("test.jsonl", f))

    print('----- File creation Done -----')
    print('----- Starting Fine Tuning Job -----')

    created_jobs = client.jobs.create(
        model="open-mistral-7b",
        training_files=[vision_mnist_train.id],
        validation_files=[vision_mnist_test.id],
        hyperparameters=TrainingParameters(
            training_steps=30,
            learning_rate=0.0001,
        ),
        integrations=[
            WandbIntegrationIn(
                project="mistral_vision",
                run_name="main",
                api_key=wandb_api_key,
            ).dict()
        ],
    )

    print(created_jobs)

    print(f'''----- All Done -> Job is Running with id {created_jobs.id} -----''')