# Mistral Vision: Exploring LLM Fine-tuning for Image Classification

## Project Overview

Mistral Vision is an experimental project that explores the potential of fine-tuning Large Language Models (LLMs) for image classification tasks. By converting images to ASCII representations, this project demonstrates an unconventional approach to adapting text-based models for visual tasks.

## How It Works

1. **Image Preprocessing**: Convert MNIST and FashionMNIST images to ASCII art.
2. **Data Formatting**: Format ASCII art into prompts for the Mistral LLM.
3. **Fine-tuning**: Fine-tune the Mistral model on the ASCII-based dataset.
4. **Inference**: Use the fine-tuned model to classify new ASCII art representations.

Below is the graph detailing the process.

```mermaid
graph TD
    classDef data fill:#FFA07A,stroke:#333,stroke-width:2px;
    classDef process fill:#98FB98,stroke:#333,stroke-width:2px;
    classDef model fill:#87CEFA,stroke:#333,stroke-width:2px;
    classDef distributed fill:#DDA0DD,stroke:#333,stroke-width:2px;
    A([MNIST/Fashion MNIST Images]):::data --> B[Split Data]:::process
    B --> DP
    subgraph DP[Distributed Processing - 40 CPU Cores]
        B --> D1[Worker 1]:::distributed
        B --> D2[Worker 2]:::distributed
        B --> D3[Worker 3]:::distributed
        B --> D4[Worker N]:::distributed
        D1 --> E1[Data Chunk 1]:::data
        D2 --> E2[Data Chunk 2]:::data
        D3 --> E3[Data Chunk 3]:::data
        D4 --> E4[Data Chunk N]:::data
        E1 --> C1[Convert to ASCII]:::process
        E2 --> C2[Convert to ASCII]:::process
        E3 --> C3[Convert to ASCII]:::process
        E4 --> C4[Convert to ASCII]:::process
        C1 --> F1[Format ASCII as Prompts]:::process
        C2 --> F2[Format ASCII as Prompts]:::process
        C3 --> F3[Format ASCII as Prompts]:::process
        C4 --> F4[Format ASCII as Prompts]:::process
    end
    F1 --> F[Combine Results]:::distributed
    F2 --> F
    F3 --> F
    F4 --> F
    F --> I[Fine-tune Mistral LLM]:::model
    I --> J[Inference with Fine-tuned Model]:::model
    J -.-> K([New image]):::data
```

## Key Components

- Image to ASCII conversion
- Data processing with multi-threading
- Integration with MistralAI's API for model fine-tuning and inference
- Handling of MNIST and FashionMNIST datasets

## Repository Structure

- `data_process.py`: Image loading, transformation, and ASCII conversion
- `process_data_before_training.py`: Data preparation for fine-tuning
- `fine_tuning.py`: Fine-tuning process management with MistralAI
- `fine_tuned_results.py`: Inference using the fine-tuned model
- `render_results.py`: Performance analysis and visualization
- `reformat_data.py`: Data reformatting utility

## Distributed Data Processing
A key feature of this project is its highly efficient data handling system. The data processing pipeline is designed to leverage multi-core processors, distributing the workload across multiple threads. This approach significantly reduces processing time, especially when dealing with large datasets like MNIST and FashionMNIST. The multi-threaded design allows for parallel processing of image conversion, ASCII transformation, and data formatting, showcasing scalable data handling techniques essential for large-scale machine learning projects.


## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mistral-vision.git
   cd mistral-vision
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Process the data:
   ```
   python data_process.py
   python process_data_before_training.py
   ```

4. Fine-tune the model:
   ```
   python fine_tuning.py
   ```

5. Run inference:
   ```
   python fine_tuned_results.py
   ```

6. Analyze results:
   ```
   python render_results.py
   ```

## Model Details

This project uses the `open-mixtral-7b` model. The fine-tuned model can be accessed via MistralAI's API using the job ID: `be2e7a1c-21e3-458a-881c-9d9adca22cef`.

## Hardware Used

Data processing was performed using NVIDIA GPUs, with computation distributed across multiple cores for efficiency.

## Contributing and Future Works

Feedback and contributions are welcome. Feel free to open an issue or submit a pull request.

- Build a new tokenizer for handling the ASCII characters more efficiently

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- MistralAI for their language models and API
- The creators of the MNIST and FashionMNIST datasets

---
