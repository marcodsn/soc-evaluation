# As the Oxford Eglish Corpus is not publicly available, we decided to generate
# text using a pre-trained language model to create a similar dataset.

import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Config
MODEL_ID = (
    "LiquidAI/LFM2.5-1.2B-Base"  # A recent language model trained on 28 trillion tokens
)
OUTPUT_FILE = "data/raw/en_texts_dynamic.txt"
TARGET_TOKENS = 512 * 2048
DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Create output directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Load model and tokenizer
print(f"Loading model on device: {DEVICE}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    dtype="bfloat16",
    # If Flash Attention 2 is available, uncomment the line below for faster generation
    # attn_implementation="flash_attention_2" if DEVICE.startswith("cuda") else None,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Print special tokens to get the BOS token
# print("Special tokens:", tokenizer.special_tokens_map)  # <|startoftext|>
# exit()

# input_ids = tokenizer("<|startoftext|>Once", return_tensors="pt").input_ids.to(
#     model.device
# )

# output = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.7,
#     min_p=0.15,
#     repetition_penalty=1.05,
#     max_new_tokens=512,
#     streamer=streamer,
# )

## Enhancing Diversity in Generated Text
# 1. Diverse Sampling Configurations
# Different settings produce different vocabulary distributions.
# Higher temp allows for rarer words; lower temp sticks to common grammar.
# https://arxiv.org/abs/2306.04140
SAMPLING_CONFIGS = [
    {"temperature": 0.85, "min_p": 0.08, "repetition_penalty": 1.1, "top_k": 50},
    {"temperature": 0.7, "min_p": 0.10, "repetition_penalty": 1.05, "top_k": 40},
    {"temperature": 0.5, "min_p": 0.15, "repetition_penalty": 1.05, "top_k": 30},
]

# 2. Category-Based Prompt Bank
# To approximate a full corpus, we need prompts that trigger specific domains.
# Base models continue the "vibe" of the prompt. We also need to screw the probability of each category,
# to prevent rare prompts to be presented too frequently.
# Weights are chosen based on rough estimates of English corpus composition (we looked at COCA and BNC corpuses).
PROMPT_CATEGORIES = {
    "narrative": {
        "weight": 0.20,  # Baseline for general English
        "prompts": [
            "<|startoftext|>The old man",
            "<|startoftext|>Suddenly,",
            "<|startoftext|>It was the best",
            "<|startoftext|>She looked at",
            "<|startoftext|>Once upon a time",
            "<|startoftext|>The door opened",
        ],
    },
    "conversational": {
        "weight": 0.20,  # For common verbs/pronouns
        "prompts": [
            "<|startoftext|>I don't think",
            "<|startoftext|>Hey, have you",
            "<|startoftext|>Well, actually",
            "<|startoftext|>Why did you",
            "<|startoftext|>Look, I just",
        ],
    },
    "news": {
        "weight": 0.20,  # Information/Facts
        "prompts": [
            "<|startoftext|>Breaking news:",
            "<|startoftext|>The government announced",
            "<|startoftext|>Yesterday, officials",
            "<|startoftext|>According to reports",
            "<|startoftext|>Markets opened",
        ],
    },
    "popular_web": {
        "weight": 0.20,  # Popular/Web content
        "prompts": [
            "<|startoftext|>Top 10",
            "<|startoftext|>I recently tried",
            "<|startoftext|>Here's how to",
            "<|startoftext|>Review:",
            "<|startoftext|>If you're looking for",
            "<|startoftext|>The best way to",
            "<|startoftext|>My experience with",
            "<|startoftext|>5 reasons why",
            "<|startoftext|>This product",
            "<|startoftext|>In this guide,",
            "<|startoftext|>You should definitely",
            "<|startoftext|>Today I want to share",
        ],
    },
    "academic": {
        "weight": 0.15,  # Lower weight to prevent jargon flooding
        "prompts": [
            "<|startoftext|>The study demonstrates",
            "<|startoftext|>Evidence suggests that",
            "<|startoftext|>However, the analysis",
            "<|startoftext|>Figure 1 shows",
        ],
    },
    "connective": {
        "weight": 0.05,  # Transition words
        "prompts": [
            "<|startoftext|>Although",
            "<|startoftext|>Despite the",
            "<|startoftext|>Furthermore,",
            "<|startoftext|>Consequently,",
        ],
    },
}


def get_category_weighted_prompt():
    """
    1. Picks a Category based on defined weights (e.g., 30% chance of Narrative).
    2. Picks a random Prompt from that category.
    """
    categories = list(PROMPT_CATEGORIES.keys())
    weights = [PROMPT_CATEGORIES[c]["weight"] for c in categories]

    # Step 1: Pick Category
    selected_cat_name = random.choices(categories, weights=weights, k=1)[0]
    selected_cat = PROMPT_CATEGORIES[selected_cat_name]

    # Step 2: Pick Prompt (Uniformly within the category)
    return random.choice(selected_cat["prompts"])


def is_high_quality(text, min_len=50):
    """
    Filters out repetitive loops or overly short garbage.
    Returns: (bool, reason)
    """
    tokens = text.split()
    total_tokens = len(tokens)

    if total_tokens < min_len:
        return False, "too_short"

    # Calculate Uniqueness Ratio (Unique words / Total words)
    # Normal English is usually 0.4 - 0.6 for short texts.
    # Loops like "The dog The dog The dog" drop this below 0.2.
    unique_tokens = len(set(tokens))
    ratio = unique_tokens / total_tokens

    if ratio < 0.35:
        return False, f"repetitive (ratio {ratio:.2f})"

    return True, "ok"


# Main Generation Loop
tot_generated = 0
start_time = time.time()

print(f"Starting generation loop. Target: {TARGET_TOKENS} tokens.")

while tot_generated < TARGET_TOKENS:
    # A. Dynamic Configuration
    prompt = get_category_weighted_prompt()
    config = random.choice(SAMPLING_CONFIGS)

    # B. Variable Lengths
    current_max_len = random.choice([256, 384, 512, 640])

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # C. Generation
    t0 = time.time()
    with torch.no_grad():  # Save VRAM
        output = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=current_max_len,
            **config,  # Unpack the selected config
        )
    t1 = time.time()

    # D. Processing & Filtering
    new_tokens_count = output.shape[1] - input_ids.shape[1]
    raw_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean text: remove newlines for line-based storage, strip whitespace
    clean_text = raw_text.replace("\n", " ").strip()

    # Check Quality
    is_valid, reason = is_high_quality(clean_text)

    if is_valid:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(clean_text + "\n")

        tot_generated += new_tokens_count
        speed = new_tokens_count / (t1 - t0)
        print(
            f"[Saved] +{new_tokens_count} toks | Total: {tot_generated} | Speed: {speed:.1f} t/s | Mode: {prompt[:20]}..."
        )
    else:
        print(f"[Skipped] {reason} | Prompt: {prompt[:20]}...")

print("Target reached. Corpus generation complete.")
