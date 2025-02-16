#!/usr/bin/env python3
import sys
import asyncio
import os
import argparse
from google import genai
from google.genai import types
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

def activate_browser_agent(steps: str) -> str:
    """Activates the browser-use agent to complete the given step by step instructions using a real browser."""
    async def run_agent():
        try:
            browser = Browser(
                config=BrowserConfig(
                    chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
                )
            )
        except Exception as e:
            print("Failed to start a new Chrome instance. Ensure that all existing Chrome instances are closed and try again.")
            raise e
        agent = Agent(task=task, llm=ChatOpenAI(model="gpt-4o"), browser=browser)
        result = await agent.run()
        input('Press Enter to close the browser...')
        await browser.close()
        return result
    return asyncio.run(run_agent())

def main():
    # Ensure the Google API key is set via the environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # Parse command line arguments for task and file to upload
    parser = argparse.ArgumentParser(description="Run browser agent with a task and file upload context")
    parser.add_argument("task", nargs='?', help="Task for the browser agent")
    parser.add_argument("--file", "-f", help="Path to file to upload")
    args = parser.parse_args()

    if not args.task:
        args.task = input("Enter the task for the browser agent: ")

    # File upload is optional; if no file path is provided, prompt the user for a PDF file.
    if not args.file:
        args.file = input("Enter the file path to upload (press Enter to skip): ")

    user_task = args.task + "\nPlease provide detailed step by step instructions for browser use."
    
    # Create the Gen AI client using the API key
    client = genai.Client(api_key=api_key)

    # File upload is optional. If a file path is provided, upload the file.
    upload_file = None
    if args.file and args.file.strip():
        upload_file = client.files.upload(file=args.file)
    
    # Generate content with function calling enabled.
    # The configuration below forces function call mode to 'ANY' so that if the model
    # deems it appropriate, it will call the provided activate_browser_agent function.
    if upload_file:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=user_task,
            config=types.GenerateContentConfig(
                tools=[activate_browser_agent],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=2),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='ANY')
                )
            )
        )
    else:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=user_task,
            config=types.GenerateContentConfig(
                tools=[activate_browser_agent],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=2),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='ANY')
                )
            )
        )
    
    # Print the result from the function call if one occurred; otherwise, print the model's response.
    if response.function_calls:
        print("Browser agent result:")
        print(response.function_calls[0].args.get('steps', ''))
    else:
        print("Response from Gen AI:")
        print(response.text)
    while True:
        followup = input("Enter follow-up message (or press Enter to exit): ")
        if not followup.strip():
            break
        followup_response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=followup,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=0),
                tools=[],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='NONE')
                )
            )
        )
        print("Follow-up response:")
        print(followup_response.text)

if __name__ == "__main__":
    main()
