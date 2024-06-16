import os
from mistralai.client import MistralClient
from mistralai.models.jobs import WandbIntegrationIn, TrainingParameters
import time

api_key = "puueDsQhS9LXPdEnooegEai7NeT4oer4"    # This key has been revoked
wandb_api_key = "062fb4bbe403b946cfaa4f52792fe7d7133c0274"  # This key has been revoked
client = MistralClient(api_key=api_key)

model_name: str = "open-mixtral-8x7b" #"open-mistral-7b" # open-mixtral-8x7b

if __name__ == "__main__":
    print('----- Starting file creation -----')
    with open("transformed_data/train.jsonl", "rb") as f:
        vision_mnist_train = client.files.create(file=("train.jsonl", f))

    with open("transformed_data/test.jsonl", "rb") as f:
        vision_mnist_test = client.files.create(file=("test.jsonl", f))

    print('----- File creation Done -----')
    print('----- Starting Fine Tuning Job -----')

    created_jobs = client.jobs.create(
        model=model_name,
        training_files=[vision_mnist_train.id],
        validation_files=[vision_mnist_test.id],
        hyperparameters=TrainingParameters(
            training_steps=100,
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

    print(f"""----- Job initialzed {created_jobs} -----""")

    retrieved_job = client.jobs.retrieve(created_jobs.id)
    while retrieved_job.status in ["RUNNING", "QUEUED"]:
        retrieved_job = client.jobs.retrieve(created_jobs.id)
        print(f""" ----- Found a running job {retrieved_job} ------- """)
        time.sleep(30)

    print(f'''----- All Done -> Job is Done with id {created_jobs.id} -----''')