import os
from pathlib import Path

os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"),
)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    max_new_tokens=40,
    do_sample=False,
    return_full_text=False,
)

# System instruction
system_instruction = "You are a helpful AI assistant. Answer clearly and simply."

# Conversation history
conversation_history = []

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Add user question to history
    conversation_history.append({
        "role": "user",
        "content": question
    })

    try:
        messages = [
            {
                "role": "system",
                "content": system_instruction
            }
        ] + conversation_history

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        result = pipe(prompt)

        answer = result[0]["generated_text"].strip()

        print("AI:", answer)

        # Save AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        print("Error:", e)