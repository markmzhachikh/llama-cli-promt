import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import warnings
import sys

# Disable warnings and set logging level
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model_and_tokenizer(model_path_or_name, hf_token=None):
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    # Load tokenizer with fast option enabled if possible
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name)

    # Move model to MPS or CPU, and use float16 for faster computation
    if torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device).half().half()  # Switch model to half precision (float16) for faster computation

    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def should_stop_generation(text):
    # Stop conditions:
    stop_sequences = ["Human:", "You:", "AI:"]
    return any(seq in text for seq in stop_sequences)


def generate_response_stream(model, tokenizer, prompt, chat_history, device):
    # Prepare input for the model
    full_prompt = f"{chat_history}\nHuman: {prompt}\nAI:"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    generated_text = ""
    max_new_tokens = 20000  # Adjust for efficiency

    # Generate response token by token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            next_word = tokenizer.decode(next_token[0])
            generated_text += next_word
            yield next_word

            if should_stop_generation(generated_text):
                break

            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM or Hugging Face model")
    parser.add_argument("--model", required=True, help="Path to local model checkpoint or name of Hugging Face model")
    parser.add_argument("--token", help="Hugging Face API token (if using a Hugging Face model)")
    args = parser.parse_args()

    # Load model, tokenizer, and set device
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.token)
    max_context_length = model.config.max_position_embeddings
    chat_history = ""

    print("Chat session started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        print("AI: ", end="", flush=True)
        response = ""
        for token in generate_response_stream(model, tokenizer, user_input, chat_history, device):
            print(token, end="", flush=True)
            response += token
        print()  # New line after response

        chat_history += f"\nHuman: {user_input}\nAI: {response}"

        # Trim chat history if it exceeds the maximum context length
        encoded_history = tokenizer.encode(chat_history)
        if len(encoded_history) > max_context_length:
            while len(encoded_history) > max_context_length:
                chat_history = chat_history.split("\n", 2)[-1]
                encoded_history = tokenizer.encode(chat_history)


if __name__ == "__main__":
    main()
