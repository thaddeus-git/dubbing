import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threading import Lock
import json_repair
import json 
from openai import OpenAI
import time
from requests.exceptions import RequestException
from core.config_utils import load_key

LOG_FOLDER = 'output/gpt_log'
LOCK = Lock()

def save_log(model, prompt, response, log_title = 'default', message = None):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_data = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "message": message
    }
    log_file = os.path.join(LOG_FOLDER, f"{log_title}.json")
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_data)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
        
def check_ask_gpt_history(prompt, model, log_title):
    # check if the prompt has been asked before
    if not os.path.exists(LOG_FOLDER):
        return False
    file_path = os.path.join(LOG_FOLDER, f"{log_title}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item["prompt"] == prompt:
                    return item["response"]
    return False

def ask_gpt(prompt, response_json=True, valid_def=None, log_title='default'):
    api_set = load_key("api")
    llm_support_json = load_key("llm_support_json")
    with LOCK:
        history_response = check_ask_gpt_history(prompt, api_set["model"], log_title)
        if history_response:
            return history_response
    
    if not api_set["key"]:
        raise ValueError(f"⚠️API_KEY is missing")
    
    messages = [{"role": "user", "content": prompt}]
    
    base_url = api_set["base_url"].strip('/') + '/v1' if 'v1' not in api_set["base_url"] else api_set["base_url"]
    client = OpenAI(api_key=api_set["key"], base_url=base_url)
    response_format = {"type": "json_object"} if response_json and api_set["model"] in llm_support_json else None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion_args = {
                "model": api_set["model"],
                "messages": messages
            }
            if response_format is not None:
                completion_args["response_format"] = response_format
                
            response = client.chat.completions.create(**completion_args)
            
            if response_json:
                try:
                    # Handle MiniMax API response which may contain thinking blocks
                    content = response.choices[0].message.content
                    # If content contains thinking blocks, extract only text
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                        content = ''.join(text_parts)
                    # Strip thinking markers from response
                    if isinstance(content, str):
                        import re
                        # Remove code block markers
                        content = re.sub(r'^[\s\n]*```\s*\n?', '', content)
                        content = re.sub(r'\n?```\s*$', '', content)
                        # Remove thinking blocks (text between thinking tags)
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                        # Remove standalone thinking labels
                        content = re.sub(r'\n?thinking\s*\n', '\n', content)
                    response_data = json_repair.loads(content)
                    
                    # check if the response is valid, otherwise save the log and raise error and retry
                    if valid_def:
                        valid_response = valid_def(response_data)
                        if valid_response['status'] != 'success':
                            save_log(api_set["model"], prompt, response_data, log_title="error", message=valid_response['message'])
                            raise ValueError(f"❎ API response error: {valid_response['message']}")
                        
                    break  # Successfully accessed and parsed, break the loop
                except Exception as e:
                    response_data = response.choices[0].message.content
                    # Handle MiniMax response format with thinking blocks
                    if isinstance(response_data, list):
                        text_parts = []
                        for block in response_data:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                        response_data = ''.join(text_parts)
                    # Strip thinking markers from string content
                    if isinstance(response_data, str):
                        import re
                        # Remove thinking block markers and content
                        response_data = re.sub(r'\n?\s*```\s*\n?', '', response_data)
                        # Extract just the JSON part if wrapped in other text
                        json_match = re.search(r'\{[^}]*"analysis"[^}]*"split"[^}]*\}', response_data, re.DOTALL)
                        if json_match:
                            response_data = json_match.group(0)
                    print(f"❎ json_repair parsing failed. Retrying: '''{response_data}'''")
                    save_log(api_set["model"], prompt, response_data, log_title="error", message=f"json_repair parsing failed.")
                    if attempt == max_retries - 1:
                        raise Exception(f"JSON parsing still failed after {max_retries} attempts: {e}\n Please check your network connection or API key or `output/gpt_log/error.json` to debug.")
            else:
                response_data = response.choices[0].message.content
                break  # Non-JSON format, break the loop directly
                
        except Exception as e:
            if attempt < max_retries - 1:
                if isinstance(e, RequestException):
                    print(f"Request error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                else:
                    print(f"Unexpected error occurred: {e}\nRetrying...")
                time.sleep(2)
            else:
                raise Exception(f"Still failed after {max_retries} attempts: {e}")
    with LOCK:
        if log_title != 'None':
            save_log(api_set["model"], prompt, response_data, log_title=log_title)

    return response_data


if __name__ == '__main__':
    # Test Minimax API
    result = ask_gpt('Respond in JSON format with {"status": "ok", "message": "hello"}', response_json=True, log_title='test_minimax')
    print("Test result:", result)