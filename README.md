# Mistral Vision

The goal of this project is to turn a Mistral LLM that is purely text generative to a vision model able to classify images.
First, the images are processed, converts them into ASCII characters, and then uses a fine-tuned model from the MistralAI model to make inferences. 

You can use the fine-tuned model using the Mistral's API querying the associated job-id.
Mistral-7b job id: be2e7a1c-21e3-458a-881c-9d9adca22cef


## Getting Started

These instructions will guide you through the process of setting up and running this project on your local machine for development and testing purposes.

### Running the Scripts

1. `process_data_before_training.py`: This script reads image files from the `transformed_data` directory, processes the images, and converts them into ASCII characters. The formatted datasets are then saved as JSONL files in the same directory.

```bash
python process_data_before_training.py
```

2. `fine_tuned_results.py`: This script reads the ASCII character files from the `transformed_data` directory, makes inferences using a fine-tuned model from the MistralAI API, and saves the results.

```bash
python fine_tuned_results.py
```