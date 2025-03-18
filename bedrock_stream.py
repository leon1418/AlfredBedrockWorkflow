import PyPDF2
import os
import sys
import time
import json
import boto3
from botocore.client import Config
from typing import Dict, Any, Optional
from urllib.parse import urlparse 

# Constants
DEFAULT_KNOWLEDGE_BASE_ID = "HKAS1IK3VY"

def get_model_mapping(aws_account: str, aws_region: str) -> Dict[str, Dict[str, str]]:
    """Generate model mapping based on AWS account and region."""
    return {
        "Deepseek-R1": {
            "actual_id": "us.deepseek.r1-v1:0",
            "arn": f"arn:aws:bedrock:{aws_region}:{aws_account}:inference-profile/us.deepseek.r1-v1:0"
        },
        "Claude-3.7": {
            "actual_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "arn": f"arn:aws:bedrock:{aws_region}:{aws_account}:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        }
    }

# Global clients for connection reuse
bedrock_client = None
bedrock_agent_client = None

def get_bedrock_client(region: str):
    """Get or create a cached Bedrock client."""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client('bedrock-runtime', region_name=region)
    return bedrock_client

def get_bedrock_agent_client(region: str):
    """Get or create a cached Bedrock agent client."""
    global bedrock_agent_client
    if bedrock_agent_client is None:
        bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
        bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=region, config=bedrock_config)
    return bedrock_agent_client

def ensure_stream_file(file_path: str) -> None:
    """Ensure the stream file exists and its directory is created."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        open(file_path, "w").close()

def write_stream_data(file_path: str, data: Dict[str, Any]) -> None:
    """Write stream data to file with proper error handling."""
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
            f.flush()
    except IOError as e:
        print(f"Error writing to stream file: {e}")
        raise

def create_base_response(model_id: str) -> Dict[str, Any]:
    """Create a base response dictionary."""
    return {
        "id": "",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": "fp_3a5770e1b4_prod0225",
        "choices": [{
            "index": 0,
            "delta": {"content": ""},
            "logprobs": None,
            "finish_reason": None
        }]
    }

def write_chunk_response(file_path: str, base_response: Dict[str, Any], content: str) -> None:
    """Write a chunk response with the given content."""
    chunk_response = base_response.copy()
    chunk_response["choices"][0]["delta"]["content"] = content
    write_stream_data(file_path, chunk_response)

def invoke_bedrock_model_stream(prompt: str, model_id: str, stream_file: str, region: str, with_file_input: bool = False) -> None:
    """Invoke Bedrock model with streaming response."""
    ensure_stream_file(stream_file)
    client = get_bedrock_client(region)
    base_response = create_base_response(model_id)
    
    try:
        messages = json.loads(prompt)

        # If with_file_input is true, write the file content in chunks
        if with_file_input:
            write_chunk_response(stream_file, base_response, "####Original File Content:\n\n")
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", [])[0].get("text", "")
                    chunk_size = 5
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        write_chunk_response(stream_file, base_response, chunk)
                        time.sleep(0.01)  # Small delay for smooth streaming
                    break
            # Add separator after file content
            write_chunk_response(stream_file, base_response, "\n####Model Response:\n\n")
            write_chunk_response(stream_file, base_response, "####Thinking:\n\n")
        else:
            write_chunk_response(stream_file, base_response, "####Thinking:\n\n")

        # print("Prompt: ", prompt)
        # Prepare parameters based on model
        if "claude-3" in model_id.lower():
            params = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "temperature": 1,
                    "maxTokens": 20000
                },
                "additionalModelRequestFields": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 16000
                    }
                }
            }
        else:
            params = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "temperature": 0.5,
                    "maxTokens": 20000,
                    "topP": 0.9
                }
            }
        
        response = client.converse_stream(**params)
        
        final_response_header_printed = False
        
        for event in response['stream']:
            #print(event)
            if 'contentBlockDelta' in event:
                chunk = event['contentBlockDelta']
                delta = chunk.get('delta', {})

                if 'reasoningContent' in delta:
                    if 'text' in delta['reasoningContent']:
                        write_chunk_response(stream_file, base_response, 
                                      delta['reasoningContent']['text'])
                elif 'text' in delta and delta['text'].strip():
                    if not final_response_header_printed:
                        write_chunk_response(stream_file, base_response, "\n\n---\n\n")
                        write_chunk_response(stream_file, base_response, "####Answer:\n\n")
                        final_response_header_printed = True
                    write_chunk_response(stream_file, base_response, delta['text'])

            if 'messageStop' in event:
                # Write final response
                chunk_response = base_response.copy()
                chunk_response["choices"][0]["delta"]["content"] = ""
                chunk_response["choices"][0]["finish_reason"] = "stop"
                write_stream_data(stream_file, chunk_response)
                with open(stream_file, "a", encoding="utf-8") as f:
                    f.write(f"data: [DONE]\n\n")
                    f.flush()  
                return

    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": type(e).__name__
            }
        }
        write_stream_data(stream_file, error_data)
        raise

def invoke_bedrock_model_stream_with_rag(prompt: str, model_id: str, model_arn: str, 
                                       knowledge_base_id: str, stream_file: str, region: str) -> None:
    """Invoke Bedrock model with RAG streaming response."""
    ensure_stream_file(stream_file)
    client = get_bedrock_agent_client(region)
    base_response = create_base_response(model_id)

    try:
        write_chunk_response(stream_file, base_response, "####Thinking:\n\n")

        response = client.retrieve_and_generate(
            input={'text': prompt},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn
                }
            }
        )

        write_chunk_response(stream_file, base_response, "\n\n---\n\n")
        write_chunk_response(stream_file, base_response, "####Answer:\n\n")
        generated_text = response['output']['text']
        chunk_size = 5

        # Stream the response in chunks
        for i in range(0, len(generated_text), chunk_size):
            write_chunk_response(stream_file, base_response, 
                               generated_text[i:i+chunk_size])
            time.sleep(0.01)

        # Write completion
        chunk_response = base_response.copy()
        chunk_response["choices"][0]["delta"]["content"] = ""
        chunk_response["choices"][0]["finish_reason"] = "stop"
        write_stream_data(stream_file, chunk_response)
        with open(stream_file, "a", encoding="utf-8") as f:
            f.write(f"data: [DONE]\n\n")
            f.flush()  
        return

    except Exception as e:
        error_data = {"error": {"message": str(e), "type": type(e).__name__}}
        write_stream_data(stream_file, error_data)
        raise

def get_file_content(file_path: str) -> str:
    """Get content from a file or URL, including PDF support."""
    try:
        # Check if the input is a URL
        parsed = urlparse(file_path)
        if parsed.scheme and parsed.netloc:  # It's a URL
            response = requests.get(file_path)
            response.raise_for_status()
            
            # If it's a PDF URL, save it temporarily and read it
            if file_path.lower().endswith('.pdf'):
                temp_pdf = "temp.pdf"
                with open(temp_pdf, 'wb') as f:
                    f.write(response.content)
                content = read_pdf(temp_pdf)
                os.remove(temp_pdf)  # Clean up
                return content
            return response.text
        else:  # It's a local file path
            if file_path.lower().endswith('.pdf'):
                return read_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
    except Exception as e:
        raise Exception(f"Failed to read file content: {str(e)}")

def read_pdf(file_path: str) -> str:
    """Read content from a PDF file."""
    try:
        content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                content.append(page.extract_text())
                
        return "\n\n".join(content)
    except Exception as e:
        raise Exception(f"Failed to read PDF file: {str(e)}")


def parse_input(user_input: str) -> str:
    """Parse and validate user input."""
    try:
        data = json.loads(user_input)
        
        if isinstance(data, list):
            transformed_messages = []
            for msg in data:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    # For assistant messages, extract specific parts
                    if msg.get("role") == "assistant" and isinstance(content, str):
                        # First check for original file content
                        file_content = ""
                        file_marker = "####Original File Content"
                        response_marker = "\n####Model Response:\n\n"
                        file_start = content.find(file_marker)
                        if file_start != -1:
                            response_start = content.find(response_marker, file_start)
                            if response_start != -1:
                                # Include the marker in the content
                                file_content = content[file_start:response_start].strip()
                        
                        # Then get the answer part
                        answer_delimiter = "\n\n---\n\n####Answer:\n\n"
                        answer_index = content.find(answer_delimiter)
                        if answer_index != -1:
                            # Include ####Answer: in the content
                            answer_content = "####Answer:\n" + content[answer_index + len(answer_delimiter):]
                            # Combine file content and answer if file content exists
                            if file_content:
                                content = f"{file_content}\n{answer_content}"
                            else:
                                content = answer_content
                    
                    # Create new message with transformed content
                    new_msg = {
                        "role": msg.get("role", ""),
                        "content": [{"text": content}]
                    }
                    transformed_messages.append(new_msg)
            
            if not transformed_messages:
                raise ValueError("No valid messages in the input")
            return json.dumps(transformed_messages)
        
        # Similar changes for the dict case...
        if isinstance(data, dict):
            if "content" not in data:
                raise ValueError("No content in the input")
            content = data["content"]
            if data.get("role") == "assistant" and isinstance(content, str):
                file_content = ""
                file_marker = "####Original File Content"
                response_marker = "\n####Model Response:\n\n"
                file_start = content.find(file_marker)
                if file_start != -1:
                    response_start = content.find(response_marker, file_start)
                    if response_start != -1:
                        file_content = content[file_start:response_start].strip()
                
                answer_delimiter = "\n\n---\n\n####Answer:\n\n"
                answer_index = content.find(answer_delimiter)
                if answer_index != -1:
                    answer_content = "####Answer:\n" + content[answer_index + len(answer_delimiter):]
                    if file_content:
                        content = f"{file_content}\n{answer_content}"
                    else:
                        content = answer_content
            
            return json.dumps([{
                "role": data.get("role", "user"),
                "content": [{"text": content}]
            }])
        
        return json.dumps([{
            "role": "user",
            "content": [{"text": str(data)}]
        }])
    
    except json.JSONDecodeError:
        return json.dumps([{
            "role": "user",
            "content": [{"text": user_input}]
        }])
    except ValueError as e:
        print(f"Inputs alert: {str(e)}, use the original inputs")
        return json.dumps([{
            "role": "user",
            "content": [{"text": user_input}]
        }])

def get_last_user_content(parsed_input: str) -> str:
    """Extract the text content from the last user message in the parsed input."""
    try:
        messages = json.loads(parsed_input)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
        return ""
    except json.JSONDecodeError:
        return ""

def get_and_update_last_user_content(parsed_input: str, final_prompt: str) -> str:
    """Replace the text content of the last user message with final_prompt."""
    try:
        messages = json.loads(parsed_input)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = [{"text": final_prompt}]
                break
        return json.dumps(messages)
    except json.JSONDecodeError:
        return parsed_input

def main():
    """Main execution function."""
    if len(sys.argv) != 9:
        print("Usage: python script.py <raw_input> <aws_account> <aws_region> <with_rag> <stream_file> <input_model_id> <knowledgeBaseId> <withFileInput>")
        sys.exit(1)

    try:
        raw_input = sys.argv[1]
        aws_account = sys.argv[2]
        aws_region = sys.argv[3]
        with_rag = sys.argv[4].lower() in ('true', '1', 'yes')
        stream_file = sys.argv[5]
        input_model_id = sys.argv[6].strip()
        knowledge_base_id = sys.argv[7].strip() or DEFAULT_KNOWLEDGE_BASE_ID
        with_file_input = sys.argv[8].lower() in ('true', '1', 'yes')  # New parameter

        # Get model mapping with dynamic AWS account and region
        MODEL_MAPPING = get_model_mapping(aws_account, aws_region)

        # final_prompt = parse_input(raw_input)
        # file_path = raw_input
        parsed_messages = parse_input(raw_input)
        # If with_file_input is true, use the parsed result as file path
        if with_file_input:
            try:
                file_path = get_last_user_content(parsed_messages)
                file_content = get_file_content(file_path)
                new_prompt = f"This is a file from user {file_path}, the content is:\n\n{file_content}"
                # Replace the last user message content with file content
                parsed_messages = get_and_update_last_user_content(parsed_messages, new_prompt)
            except Exception as e:
                raise Exception(f"Failed to process file input: {str(e)}")
        
        if not parsed_messages.strip():
            raise ValueError("Input can't be null")

        if input_model_id not in MODEL_MAPPING:
            raise ValueError(f"Invalid modelId: {input_model_id}. Valid options are: {', '.join(MODEL_MAPPING.keys())}")

        processed_model = MODEL_MAPPING[input_model_id]
        model_id = processed_model["actual_id"]
        model_arn = processed_model["arn"]

        if with_rag:
            invoke_bedrock_model_stream_with_rag(get_last_user_content(parsed_messages), model_id, model_arn, 
                                               knowledge_base_id, stream_file, aws_region)
        else:
            invoke_bedrock_model_stream(parsed_messages, model_id, stream_file, aws_region, with_file_input)
    except Exception as e:
        print(f"Failed: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()
