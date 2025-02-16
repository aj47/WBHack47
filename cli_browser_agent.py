#!/usr/bin/env python3
import sys
import asyncio
import os
import argparse
import json
from datetime import datetime
from google import genai
from google.genai import types
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from query_image_text_pdf import search_pdf
import weave
from together import Together
from typing import Any, Optional, Dict, List, Literal
from pydantic import Field, BaseModel, ValidationError


weave.init('metis')

@weave.op()
def activate_browser_agent(together_client, steps: str, task: str) -> str:
    """Activates the browser-use agent to complete the given step by step instructions using a real browser."""
    print("Executing browser steps:", steps)
    
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
        agent = Agent(task=steps, llm=ChatOpenAI(model="gpt-4o"), browser=browser)
        result = await agent.run()
        input('Press Enter to close the browser...')
        await browser.close()
        return result

    result = asyncio.run(run_agent())
    
    # Get feedback after browser use
    current_call = weave.require_current_call()
    # Create and save log entry
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": str(task),
            "steps": str(steps),
            "result": str(result)
        }
        
        while True:
            feedback = input("Was this helpful? (y/n): ").strip().lower()
            if feedback == 'y':
                current_call.feedback.add_reaction("👍")
                break
            elif feedback == 'n':
                current_call.feedback.add_reaction("👎")
                break
            print("Please enter either y or n")
        
        comment = input("Any additional comments? (press Enter to skip): ").strip()
        if comment:
            current_call.feedback.add_note(comment)
        
        # Add feedback to log entry
        log_entry["feedback"] = "positive" if feedback == 'y' else "negative"
        log_entry["comment"] = str(comment) if comment else ""

        # LLM Evaluation
        evaluation, feedback = single_eval(together_client, task, steps, str(result), log_entry["feedback"], log_entry["comment"])

        # Adding a note
        current_call.feedback.add_note(f"LLM Reflection: {feedback}")

        # Adding custom key/value pairs.
        current_call.feedback.add("llm_evaluation", { "value": {evaluation} })
        
        # Append to log file
        log_file = "browser_agent_logs.json"
        logs = []
        
        # Read existing logs if file exists
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                logs = []
        
        # Ensure logs is a list
        if not isinstance(logs, list):
            logs = []
            
        logs.append(log_entry)
        
        # Write updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Failed to save log: {e}")
    
    return result

def get_user_input() -> str:
    """Get task from user input."""
    parser = argparse.ArgumentParser(description="Run browser agent with a task")
    parser.add_argument("task", nargs='?', help="Task for the browser agent")
    args = parser.parse_args()

    if not args.task:
        args.task = input("Enter the task for the browser agent: ")

    return args.task

@weave.op
def call_gemini(client: genai.Client, user_task: str, file_path: str | None) -> types.GenerateContentResponse:
    """Make Gemini API call with the given task and optional file path."""
    # File upload is optional. If a file path is provided, upload the file.
    upload_file = None
    if file_path and file_path.strip():
        upload_file = client.files.upload(file=file_path)
    # Define the function declaration for activate_browser_agent
    function = types.FunctionDeclaration(
        name='activate_browser_agent',
        description='Activates the browser-use agent to complete the given step by step instructions using a real browser',
        parameters=types.Schema(
            type='OBJECT',
            properties={
                'steps': types.Schema(
                    type='STRING',
                    description='Detailed step by step instructions for browser use. The PDF file is already available to the agent - do NOT ask for a URL.',
                ),
            },
            required=['steps'],
        ),
    )

    tool = types.Tool(function_declarations=[function])

    # Generate content with function calling enabled
    # Create content parts including both text and file
    content_parts = [genai.types.Part(text=(user_task + "\nPlease provide detailed step by step instructions for browser use."))]
    if upload_file:
        content_parts.append(genai.types.Part(file_data=genai.types.FileData(
            mime_type=upload_file.mime_type,
            file_uri=upload_file.uri
        )))
    
    return client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=content_parts,
        config=types.GenerateContentConfig(
            tools=[tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=2),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='ANY')
            )
        )
    )


# Simple JSON mode LLM call helper function - will be used by the Evaluator
def JSON_llm(together_client, user_prompt : str, schema : BaseModel, system_prompt : Optional[str] = None):
    """ Run a language model with the given user prompt and system prompt, and return a structured JSON object. """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        
        extract = together_client.chat.completions.create(
            messages=messages,
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            response_format={
                "type": "json_object",
                "schema": schema.model_json_schema(),
            },
        )
        
        response = json.loads(extract.choices[0].message.content)
        return response
        
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {str(e)}")
    
EVALUATOR_PROMPT = """Evaluate this following trace for an agent executing a task. You should be evaluate whether or not it was successful at solving the task.
If it was not successful, then explain what stage it failed in the execution trace.
Only output "PASS" if all criteria are met and you have no further suggestions for improvements, otherwise output "FAIL".
Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.
Only output JSON.

Task: {task}
Steps: {steps}
Trace: {trace}
Feedback: {feedback}
Comment: {comment}

Output:"""

def evaluate(together_client, schema, task, steps, trace, feedback, comment) -> tuple[str, str]:
    """Evaluate if a solution meets requirements."""
    full_prompt = EVALUATOR_PROMPT.format(task=task, steps=steps, trace=trace, feedback=feedback, comment=comment)
    
    response = JSON_llm(together_client, full_prompt, schema)
    
    evaluation = response["evaluation"]
    feedback = response["feedback"]

    print("=== EVALUATION START ===")
    print(f"Status: {evaluation}")
    print(f"Feedback: {feedback}")
    print("=== EVALUATION END ===\n")

    return evaluation, feedback

def single_eval(together_client, task, steps, trace, feedback, comment) -> tuple[str, list[dict]]:
    #Build a schema for the evaluation
    class Evaluation(BaseModel):
        evaluation: Literal["PASS", "NEEDS_IMPROVEMENT", "FAIL"]
        feedback: str

    # While the generated response is not passing, keep generating and evaluating
    return evaluate(together_client, Evaluation, task, steps, trace, feedback, comment)


@weave.op()
def main():
    # Ensure the Google API key is set via the environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        print("Error: Please set the TOGETHER_API_KEY environment variable.")
        sys.exit(1)
    together_client = Together(api_key= together_api_key)
    
    # Get user input and find relevant PDF
    user_task = get_user_input()
    search_result = search_pdf(user_task)
    file_path = search_result['pdf_name']+".pdf" if search_result else None
    
    # Create the Gen AI client using the API key
    client = genai.Client(api_key=api_key)

    # Call Gemini with weave attributes
    with weave.attributes({'user_intent': user_task, 'doc_file': file_path}):
        response = call_gemini(client, user_task, file_path)
    
    # Execute the browser agent if we get steps
    if response.function_calls:
        steps = response.function_calls[0].args.get('steps', '')
        print("Browser agent result:")
        print(steps)
        # Actually execute the browser automation
        activate_browser_agent(together_client, steps, user_task)
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
