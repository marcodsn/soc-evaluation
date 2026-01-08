# As the Oxford Eglish Corpus is not publicly available, we decided to generate
# text using a pre-trained language model to create a similar dataset.
# https://pubmed.ncbi.nlm.nih.gov/41169687/

import random
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = (
    "LiquidAI/LFM2.5-1.2B-Base"  # A recent language model trained on 28 trillion tokens
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="mps",
    dtype="bfloat16",
    #   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
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

input_prompts = [
    "<|startoftext|>One time",
    "<|startoftext|>The",
    "<|startoftext|>In",
    "<|startoftext|>So,",  # Just "so" made the model want to generate french text ("soit")
    "<|startoftext|>At",
    "<|startoftext|>And",
]

tot_generated = 0
target_min = 512 * 32

while tot_generated < target_min:
    random_prompt = random.choice(input_prompts)

    input_ids = tokenizer(random_prompt, return_tensors="pt").input_ids.to(model.device)

    st = time.time()
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.5,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    et = time.time()

    generated_tokens = output.shape[1] - input_ids.shape[1]
    tot_generated += generated_tokens

    with open("data/raw/en_texts.txt", "a", encoding="utf-8") as f_out:
        generated_text = (
            tokenizer.decode(output[0, :], skip_special_tokens=True)
            .replace("\n", " ")
            .strip()
        )
        f_out.write(generated_text + "\n")

    print(
        f"Generated {generated_tokens} tokens, total so far: {tot_generated}/{target_min}. Current speed: {generated_tokens / (et - st):.2f} tok/s"
    )
