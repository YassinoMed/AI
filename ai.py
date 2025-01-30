import subprocess
import sys
import logging
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Set up the environment with correct package versions
    """
    logger.info("Setting up environment with correct package versions...")
    
    requirements = [
        "typing-extensions>=4.5.0",
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.13.0",
        "colorama",
        "rich"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        for req in requirements:
            logger.info(f"Installing {req}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", req])
        logger.info("Environment setup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
    top_p: float = 0.9,
) -> List[str]:
    """
    Generate text using the loaded model.
    """
    try:
        # Encode the prompt without padding
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,  # Use EOS token as pad token
            )

        # Decode and return the generated sequences
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return generated_texts

    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return []

def chat_with_bot(model, tokenizer):
    """
    Interact with the chatbot in a loop.
    """
    logger.info("Starting chatbot. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            logger.info("Exiting chatbot.")
            break
        
        try:
            generated_texts = generate_text(
                model,
                tokenizer,
                user_input,
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
            
            for i, text in enumerate(generated_texts, 1):
                logger.info(f"\nBot: {text}")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

def main():
    """
    Main function with environment setup and chatbot interaction
    """
    if not setup_environment():
        logger.error("Failed to set up environment. Exiting.")
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        logger.error(f"Failed to import required packages: {str(e)}")
        return

    model_name = "tiiuae/falcon-40b-instruct"
    hf_token = "YOUR_Token"  # Replace with your token

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token if hf_token != "YOUR_Token" else None
        )

        # Set the pad token to be the same as the EOS token
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            token=hf_token if hf_token != "YOUR_Token" else None
        )

        logger.info("Model loaded successfully!")
        
        # Start chatting with the bot
        chat_with_bot(model, tokenizer)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()