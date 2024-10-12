import argparse
import logging
import sys
import warnings
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress transformers warnings
transformers.logging.set_verbosity_error()


def load_model_and_tokenizer(model_path_or_name, hf_token=None):
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        device_map='auto',
        # torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def should_stop_generation(text):
    if text.endswith('\n'):
        if check_for_repetitions(text, 2):
            return True
    stop_sequences = ["Human:", "You:", "AI:"]
    return any(seq in text for seq in stop_sequences)


def check_for_repetitions(response: str, max_repetitions: int = 2) -> bool:
    """
    Checks for excessive repetition within a generated response.

    Parameters:
    response (str): The generated response from the chatbot.
    max_repetitions (int): The maximum number of allowed repetitions for a single line.

    Returns:
    bool: True if repetition exceeds the allowed threshold, otherwise False.
    """
    # Split the response into lines
    lines = response.strip().split('\n')

    # Dictionary to count occurrences of each line
    line_counts = {}

    # Count each line's occurrence
    for line in lines:
        if line.strip() == "":
            continue  # Skip empty lines
        line_counts[line] = line_counts.get(line, 0) + 1
        # If any line exceeds the repetition threshold, return True
        if line_counts[line] > max_repetitions:
            return True

    # No excessive repetition detected
    return False

@torch.inference_mode()
def generate_response_stream(model, tokenizer, prompt, chat_history, temperature=1.0):
    full_prompt = f"{chat_history}\nHuman: {prompt}\nAI:"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

    generated_text = ""
    max_new_tokens = 1000

    past_key_values = None
    for _ in range(max_new_tokens):
        try:
            with torch.no_grad():
                if past_key_values is not None:
                    past_key_values = tuple(tuple(p.to(model.device) for p in pkvs) for pkvs in past_key_values)

                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :] / temperature  # Scale logits by temperature
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                past_key_values = outputs.past_key_values

                input_ids = next_token

                next_word = tokenizer.decode(next_token[0])
                generated_text += next_word
                yield next_word

                if should_stop_generation(generated_text):
                    break
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            break


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM or Hugging Face model")
    parser.add_argument("--model", required=True, help="Path to local model checkpoint or name of Hugging Face model")
    parser.add_argument("--token", help="Hugging Face API token (if using a Hugging Face model)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for response generation")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.token)
    max_context_length = model.config.max_position_embeddings
    chat_history = ""

    print("Chat session started. Type 'quit' to exit. Type Ctrl+D to confirm input.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        print("AI: ", end="", flush=True)
        response = ""
        try:
            for token in generate_response_stream(model, tokenizer, user_input, chat_history, temperature=args.temperature):
                print(token, end="", flush=True)
                response += token
            print()  # New line after response
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")

        chat_history += f"\nHuman: {user_input}\nAI: {response}"

        # Trim chat history if it exceeds the maximum context length
        encoded_history = tokenizer.encode(chat_history)
        if len(encoded_history) > max_context_length:
            while len(encoded_history) > max_context_length:
                chat_history = chat_history.split("\n", 2)[-1]
                encoded_history = tokenizer.encode(chat_history)


if __name__ == "__main__":
    main()
