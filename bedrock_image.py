import boto3
import base64
import json
from io import BytesIO
import sys
from PIL import Image
from typing import Optional

# Global client for connection reuse
bedrock_client = None

def get_bedrock_client(region: str):
    """Get or create a Bedrock client using connection pooling."""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=region)
    return bedrock_client

def validate_image_params(height: int, width: int, cfg_scale: float, steps: int) -> None:
    """Validate image generation parameters."""
    if not (256 <= height <= 1024 and height % 64 == 0):
        raise ValueError("Height must be between 256 and 1024 and divisible by 64")
    if not (256 <= width <= 1024 and width % 64 == 0):
        raise ValueError("Width must be between 256 and 1024 and divisible by 64")
    if not (1.0 <= cfg_scale <= 20.0):
        raise ValueError("cfg_scale must be between 1.0 and 20.0")
    if not (10 <= steps <= 150):
        raise ValueError("steps must be between 10 and 150")

def generate_image_with_bedrock(
    prompt: str,
    aws_region: str,
    output_path: str = "generated_image.png",
    model_id: str = "stability.stable-diffusion-xl-v1",
    seed: Optional[int] = None,
    style_preset: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    cfg_scale: float = 8.0,
    steps: int = 50
) -> None:
    """
    Generate an image using Amazon Bedrock.
    
    Args:
        prompt (str): The text prompt for image generation
        aws_region (str): AWS region to use for Bedrock client
        output_path (str): Path where the generated image will be saved
        model_id (str): The Bedrock model ID to use
        seed (Optional[int]): Seed for reproducible generation
        style_preset (Optional[str]): Style preset to use
        height (int): Image height (must be between 256 and 1024, divisible by 64)
        width (int): Image width (must be between 256 and 1024, divisible by 64)
        cfg_scale (float): Configuration scale (between 1.0 and 20.0)
        steps (int): Number of inference steps (between 10 and 150)
    
    Raises:
        ValueError: If any parameters are invalid
        Exception: For any other errors during generation
    """
    try:
        # Validate parameters
        validate_image_params(height, width, cfg_scale, steps)
        
        # Get or create client with specified region
        bedrock = get_bedrock_client(aws_region)
        
        # Prepare request body
        body = {
            "text_prompts": [{"text": prompt, "weight": 1.0}],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "steps": steps
        }

        # Add optional parameters if provided
        body.update({k: v for k, v in {
            "seed": seed,
            "style_preset": style_preset
        }.items() if v is not None})
        
        # Make API call
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        # Process response and save image
        response_body = json.loads(response["body"].read())
        base64_image = response_body["artifacts"][0]["base64"]
        
        # Use context managers for automatic resource cleanup
        with BytesIO(base64.b64decode(base64_image)) as bio:
            with Image.open(bio) as image:
                image.save(output_path, optimize=True)
        
        print(f"Picture saved to: {output_path}")

    except Exception as e:
        print(f"Generate picture failed: {str(e)}")
        raise

def parse_seed(seed_str: str) -> Optional[int]:
    """Parse seed string to integer or None."""
    if seed_str.lower() == "null":
        return None
    try:
        return int(seed_str)
    except ValueError:
        raise ValueError(f"Invalid seed value: {seed_str}. Must be integer or 'null'")

def parse_style_preset(style_preset_str: str) -> Optional[str]:
    """Parse style preset string."""
    return None if style_preset_str.lower() == "none" else style_preset_str

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 7:
        print("Usage: python script.py '<prompt>' '<aws_region>' '<file_path>' '<model_id>' '<seed_str>' '<style_preset_str>'")
        sys.exit(1)

    try:
        # Parse and validate command line arguments
        raw_input = sys.argv[1].strip()
        aws_region = sys.argv[2].strip()
        file_path = sys.argv[3].strip()
        model_id = sys.argv[4].strip()
        seed_str = sys.argv[5].strip()
        style_preset_str = sys.argv[6].strip()
        
        # Validate required inputs
        if not raw_input:
            raise ValueError("Prompt cannot be empty")
        if not file_path:
            raise ValueError("File path cannot be empty")
        if not model_id:
            raise ValueError("Model ID cannot be empty")
        if not aws_region:
            raise ValueError("AWS region cannot be empty")
            
        # Parse optional parameters
        seed = parse_seed(seed_str)
        style_preset = parse_style_preset(style_preset_str)

        # Generate image
        generate_image_with_bedrock(
            prompt=raw_input,
            aws_region=aws_region,
            output_path=file_path,
            model_id=model_id,
            seed=seed,
            style_preset=style_preset
        )

    except Exception as e:
        print(f"Failed: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()
