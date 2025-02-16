# Stytch Platform Assistant with PDF-Guided Automation

This project implements a command-line interface that helps users navigate and accomplish tasks in the Stytch platform. It leverages Google's Gen AI SDK and a browser automation agent to execute detailed step-by-step instructions. The tool uses ApertureDB to intelligently retrieve relevant documentation from Stytch's PDF guides to inform the automated processes.

## Features

- **Interactive Assistance:** Engage in a conversation-like interface to specify your Stytch platform tasks.
- **Smart Documentation Lookup:** Uses ApertureDB to automatically find and reference relevant sections from Stytch's documentation PDFs.
- **Automated Task Execution:** Leverages browser automation to actually perform the tasks in the Stytch platform.
- **Google Gemini AI Integration:** Utilizes the [google-genai](https://github.com/googleapis/python-genai) SDK to understand tasks and generate precise automation instructions.
- **Execution Feedback:** Provides real-time feedback on task execution and allows for follow-up questions.
  
## Requirements

- Python 3.7 or later
- [google-genai](https://github.com/googleapis/python-genai) (`pip install google-genai`)
- [browser-use](https://github.com/browser-use/browser-use) (`pip install browser-use`)
- [langchain_openai](https://github.com/langchain-ai/langchain) (`pip install langchain_openai`)
- [aperturedb](https://github.com/aperture-data/aperturedb-python) (`pip install aperturedb`)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (`pip install sentence-transformers`)

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

Run the assistant:
```bash
python cli_browser_agent.py
```

You will be prompted to enter your desired Stytch platform task. The assistant will:
1. Search through Stytch's documentation using ApertureDB to find relevant guidance
2. Generate precise step-by-step instructions
3. Execute the task automatically in your browser
4. Provide feedback and allow for follow-up questions

Example tasks:
- "Create a new organization in Stytch"
- "Add a new user to my Stytch project"
- "Configure authentication settings"

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
