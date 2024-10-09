import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model_and_tokenizer(model_path_or_name, hf_token=None):
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name, device_map="auto")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, chat_history, max_new_tokens):
    full_prompt = f"{chat_history}\nHuman: {prompt}\nAI:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("AI:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM or Hugging Face model")
    parser.add_argument("--model", required=True, help="Path to local model checkpoint or name of Hugging Face model")
    parser.add_argument("--token", help="Hugging Face API token (if using a Hugging Face model)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.token)
    max_context_length = model.config.max_position_embeddings
    chat_history = ""

    print("Chat session started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = generate_response(model, tokenizer, user_input, chat_history, max_new_tokens=500)
        print(f"AI: {response}")

        chat_history += f"\nHuman: {user_input}\nAI: {response}"

        # Trim chat history if it exceeds the maximum context length
        encoded_history = tokenizer.encode(chat_history)
        if len(encoded_history) > max_context_length:
            # Remove earliest messages until it fits
            while len(encoded_history) > max_context_length:
                chat_history = chat_history.split("\n", 2)[-1]
                encoded_history = tokenizer.encode(chat_history)


if __name__ == "__main__":
    main()