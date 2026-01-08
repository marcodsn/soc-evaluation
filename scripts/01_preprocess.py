# Since the raw data contains a complex structure and various tags,
# we use this python script to simplify and clean the data for our
# specific tasks before loading in R.

import csv
import json
import os
import re

# Paths
input_path = os.path.join("external", "SOC-2508", "data.jsonl")
output_path = os.path.join("data", "processed", "data.csv")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)


def clean_text(text):
    # Remove <media_type>...</media_type> content entirely (descriptions aren't dialogue and can contain more "not-unique" text)
    text = re.sub(r"<(image|gif|audio).*?>.*?</\1>", "", text, flags=re.DOTALL)
    # Remove standalone tags like <delay/> or <end/>
    text = re.sub(r"<[^>]+>", "", text)
    # Add spaces around em-dashes and ellipses so they don't combine words (though we keep hyphens as they are common in words)
    text = text.replace("—", " - ")  # Em-dash
    text = text.replace("–", " - ")  # En-dash
    text = text.replace("…", " ... ")  # Unicode Ellipsis
    text = text.replace("...", " ... ")  # Standard Ellipsis
    # We keep only alphanumeric chars, basic punctuation, and spaces
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"\-\s]", "", text)
    # Remove extra whitespace
    return " ".join(text.split())


print("Processing data...")

with (
    open(input_path, "r", encoding="utf-8") as f_in,
    open(output_path, "w", newline="", encoding="utf-8") as f_out,
):
    writer = csv.writer(f_out)
    # Header: We keep chat_id and traits for grouping later
    writer.writerow(["chat_id", "persona1_traits", "persona2_traits", "topic", "text"])

    for line in f_in:
        try:
            entry = json.loads(line)

            # 1. Metadata Extraction
            exp = entry.get("experience", {})
            p1_traits = ",".join(exp.get("persona1", {}).get("traits", []))
            p2_traits = ",".join(exp.get("persona2", {}).get("traits", []))
            topic = exp.get("topic", "")

            # 2. Text Aggregation
            full_conversation = []
            for part in entry.get("chat_parts", []):
                for msg in part.get("messages", []):
                    clean_msg = clean_text(msg)
                    if clean_msg:
                        full_conversation.append(clean_msg)

            # 3. Write to CSV
            writer.writerow(
                [
                    entry["chat_id"],
                    p1_traits,
                    p2_traits,
                    topic,
                    " ".join(full_conversation),
                ]
            )

        except json.JSONDecodeError:
            print("Skipping invalid JSON line")

print(f"Done! Clean data saved to {output_path}")
