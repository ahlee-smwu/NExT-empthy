"""Utility script to prepare empathy training data from public datasets.

This script converts publicly available empathy/emotion datasets into the
format required by the NExT-empthy multimodal empathy model.

Supported source formats:
    - EmpatheticDialogues (Facebook Research)
    - Custom CSV format

Usage:
    python data/empathy/prepare_empathy_data.py \
        --source_format empatheticdialogues \
        --input_path /path/to/empatheticdialogues/train.csv \
        --output_path ./data/empathy/empathy_text_instruction.json

    python data/empathy/prepare_empathy_data.py \
        --source_format custom_csv \
        --input_path /path/to/custom_data.csv \
        --output_path ./data/empathy/empathy_text_instruction.json \
        --emotion_col emotion \
        --context_col situation \
        --utterance_col utterance \
        --response_col response
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from nextgpt.constants import EMOTION_LABELS


def convert_empatheticdialogues(input_path: str, output_path: str):
    """Convert EmpatheticDialogues dataset to empathy training format.

    EmpatheticDialogues CSV format:
        conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags

    The 'context' column contains the emotion label.
    """
    conversations_by_id = defaultdict(lambda: {"emotion": None, "turns": []})

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row.get("conv_id", "").strip()
            if not conv_id:
                continue

            emotion = row.get("context", "").strip().lower().replace(" ", "")
            utterance = row.get("utterance", "").strip().replace("_comma_", ",")
            speaker_idx = int(row.get("speaker_idx", 0))

            if conversations_by_id[conv_id]["emotion"] is None:
                conversations_by_id[conv_id]["emotion"] = emotion

            role = "human" if speaker_idx == 0 else "gpt"
            conversations_by_id[conv_id]["turns"].append(
                {"from": role, "value": utterance}
            )

    # Convert to output format
    output_data = []
    for conv_id, conv in conversations_by_id.items():
        if len(conv["turns"]) < 2:
            continue

        # Ensure conversation starts with human and alternates
        turns = conv["turns"]
        cleaned_turns = []
        expected_role = "human"
        for turn in turns:
            if turn["from"] == expected_role:
                cleaned_turns.append(turn)
                expected_role = "gpt" if expected_role == "human" else "human"

        if len(cleaned_turns) < 2:
            continue

        # Ensure even number of turns (complete pairs)
        if len(cleaned_turns) % 2 != 0:
            cleaned_turns = cleaned_turns[:-1]

        sample = {
            "emotion": conv["emotion"],
            "conversations": cleaned_turns,
        }
        output_data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Converted {len(output_data)} conversations to {output_path}")
    print(f"Emotion distribution:")
    emotion_counts = defaultdict(int)
    for sample in output_data:
        emotion_counts[sample["emotion"]] += 1
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count}")


def convert_custom_csv(
    input_path: str,
    output_path: str,
    emotion_col: str = "emotion",
    context_col: str = "situation",
    utterance_col: str = "utterance",
    response_col: str = "response",
):
    """Convert custom CSV format to empathy training format.

    Expected CSV columns: emotion, situation/utterance (human input), response (model output)
    """
    output_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = row.get(emotion_col, "").strip().lower()

            human_msg = row.get(context_col, "").strip()
            if utterance_col != context_col and utterance_col in row:
                utterance = row.get(utterance_col, "").strip()
                if utterance:
                    human_msg = f"{human_msg} {utterance}" if human_msg else utterance

            response = row.get(response_col, "").strip()

            if not human_msg or not response:
                continue

            sample = {
                "emotion": emotion,
                "conversations": [
                    {"from": "human", "value": human_msg},
                    {"from": "gpt", "value": response},
                ],
            }
            output_data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Converted {len(output_data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare empathy training data from public datasets"
    )
    parser.add_argument(
        "--source_format",
        type=str,
        required=True,
        choices=["empatheticdialogues", "custom_csv"],
        help="Source dataset format",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output JSON file"
    )
    parser.add_argument(
        "--emotion_col",
        type=str,
        default="emotion",
        help="Column name for emotion label (custom_csv only)",
    )
    parser.add_argument(
        "--context_col",
        type=str,
        default="situation",
        help="Column name for context/situation (custom_csv only)",
    )
    parser.add_argument(
        "--utterance_col",
        type=str,
        default="utterance",
        help="Column name for user utterance (custom_csv only)",
    )
    parser.add_argument(
        "--response_col",
        type=str,
        default="response",
        help="Column name for empathetic response (custom_csv only)",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.source_format == "empatheticdialogues":
        convert_empatheticdialogues(args.input_path, args.output_path)
    elif args.source_format == "custom_csv":
        convert_custom_csv(
            args.input_path,
            args.output_path,
            args.emotion_col,
            args.context_col,
            args.utterance_col,
            args.response_col,
        )


if __name__ == "__main__":
    main()
