# Gen AI Chat Interface with Local PDF Context

This project implements a command-line chat interface that leverages the Google Gen AI SDK and a Browser Use agent to generate detailed step-by-step instructions for accomplishing user-specified tasks. It also supports incorporating context from a local PDF file to inform the generated instructions.

## Features

- **Interactive Chat Interface:** Engage in a chat-like conversation where you can continuously enter tasks.
- **Local PDF Integration:** Optionally upload a PDF file to enrich the context for task instructions.
- **Google Gemini API Integration:** Utilizes the [google-genai](https://github.com/googleapis/python-genai) SDK for content generation.
- **Detailed Instructions:** Generates step-by-step guidance on how to achieve your task using a browser automation agent.
  
## Requirements

- Python 3.7 or later
- [google-genai](https://github.com/googleapis/python-genai) (`pip install google-genai`)
- [browser-use](https://github.com/browser-use/browser-use) (`pip install browser-use`)
- [langchain_openai](https://github.com/langchain-ai/langchain) (`pip install langchain_openai`)

## Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**
   ```bash
   pip install google-genai browser-use langchain_openai
   ```

3. **Set the API Key:**
   Ensure you have your Gemini API key and set it as an environment variable:
   ```bash
   export GEMINI_API_KEY='your_api_key_here'
   ```

## Usage

Run the CLI script:
```bash
python cli_browser_agent.py
```

You will then be prompted to:
- **Enter the task:** Type your desired task.
- **Enter the path for a local PDF file:** Optionally, provide a path to a PDF file to include in the context (or leave blank to skip).

The system will generate and display detailed step-by-step instructions on how to achieve your task using the browser agent.

## Customization

- **Model Configuration:** The model used for text generation is set to `"gemini-2.0-flash-001"`. Modify this in `cli_browser_agent.py` if needed.
- **System Instructions:** The system prompts given to the model can be adjusted to change how detailed the instructions are.
- **Functionality:** The script can be extended with additional tools or integrations as needed.

## Troubleshooting

- **Missing API Key:** Ensure the `GEMINI_API_KEY` environment variable is correctly set.
- **File Upload Issues:** Verify the provided PDF file path is correct and the file is accessible.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai)
- [Browser Use](https://github.com/browser-use/browser-use)
- [LangChain OpenAI](https://github.com/langchain-ai/langchain)
