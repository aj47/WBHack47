#!/usr/bin/env python3
import sys
import asyncio
import os
from google import genai
from google.genai import types
from browser_use import Agent
from langchain_openai import ChatOpenAI

def activate_browser_agent(task: str) -> str:
    """Activates the browser-use agent to complete the given task."""
    async def run_agent():
        agent = Agent(task=task, llm=ChatOpenAI(model="gpt-4o"))
        result = await agent.run()
        return result
    return asyncio.run(run_agent())

def main():
    # Ensure the Google API key is set via the environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # Create the Gen AI client using the API key
    client = genai.Client(api_key=api_key)
    
    print("Welcome to the Gen AI Chat Interface. Type 'exit' to quit.")
    while True:
        user_task = input("Enter the task: ").strip()
        if user_task.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        pdf_path = input("Enter the path for a local PDF file to include in context (or leave blank to skip): ").strip()
        if pdf_path:
            try:
                uploaded_pdf = client.files.upload(file=pdf_path)
                contents = [user_task, uploaded_pdf]
                system_instruction_text = "Based on the attached PDF, generate detailed step-by-step instructions for how to achieve the task using a browser agent."
            except Exception as e:
                print(f"Warning: Could not upload file: {e}")
                contents = user_task
                system_instruction_text = "Generate detailed step-by-step instructions for how to achieve the task using a browser agent."
        else:
            contents = user_task
            system_instruction_text = "Generate detailed step-by-step instructions for how to achieve the task using a browser agent."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction_text,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=2),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='ANY')
                )
            )
        )
        
        print("Response from Gen AI:")
        print(response.text)

if __name__ == "__main__":
    main()
