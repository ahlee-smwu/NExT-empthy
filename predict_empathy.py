"""Inference script for the NExT-empthy multimodal empathy model.

This script loads a fine-tuned empathy model and generates empathetic responses
to multimodal inputs (text, image, video, audio).

Usage:
    # Text-only empathetic response
    python predict_empathy.py \
        --model_path ./checkpoints/empathy_finetune_1 \
        --prompt "I just lost my job and I feel terrible about it."

    # With image input
    python predict_empathy.py \
        --model_path ./checkpoints/empathy_finetune_1 \
        --prompt "This is how I feel today." \
        --image ./path/to/emotional_image.jpg

    # With audio input
    python predict_empathy.py \
        --model_path ./checkpoints/empathy_finetune_1 \
        --prompt "Listen to my voice, I'm not doing well." \
        --audio ./path/to/voice_recording.wav
"""

import argparse
import os

import torch
import transformers
from PIL import Image

from nextgpt.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    EMOTION_LABELS,
    IMAGE_TOKEN_INDEX,
)
from nextgpt.conversation import conv_templates, SeparatorStyle
from nextgpt.model.builder import load_pretrained_model
from nextgpt.utils import disable_torch_init
from nextgpt.mm_utils import tokenizer_image_token, tokenizer_multiple_token

from predict import GenerateArguments, StoppingCriteriaSub, load_image

from transformers import StoppingCriteriaList


def setup_model(model_path, model_base=None, load_8bit=False, load_4bit=False):
    """Load the fine-tuned empathy model."""
    disable_torch_init()
    model_name = os.path.basename(model_path)
    tokenizer, model, image_processor, video_processor, audio_processor, context_len, model_config = (
        load_pretrained_model(
            model_path, model_base, model_name,
            load_8bit=load_8bit, load_4bit=load_4bit
        )
    )
    return tokenizer, model, image_processor, video_processor, audio_processor, context_len, model_config


def generate_empathetic_response(
    tokenizer, model, model_config,
    image_processor=None,
    prompt="",
    image_path=None,
    video_path=None,
    audio_path=None,
    conv_mode="empathy_v1",
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
):
    """Generate an empathetic response given multimodal inputs.

    Args:
        tokenizer: The tokenizer.
        model: The fine-tuned empathy model.
        model_config: Model configuration.
        image_processor: Image preprocessor.
        prompt: Text input from user.
        image_path: Optional path to an image file.
        video_path: Optional path to a video file.
        audio_path: Optional path to an audio file.
        conv_mode: Conversation template to use.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        str: The empathetic response text.
    """
    # Prepare generation arguments
    parser = transformers.HfArgumentParser(GenerateArguments)
    generation_args = parser.parse_args_into_dataclasses(args=[])[0]
    generation_args.temperature = temperature
    generation_args.top_p = top_p
    generation_args.max_new_tokens = max_new_tokens

    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=[[835]], encounters=1)]
    )
    generation_args.stopping_criteria = stopping_criteria

    # Build conversation
    conv = conv_templates[conv_mode].copy()

    # Prepend modality tokens based on provided inputs
    input_text = prompt
    if image_path is not None:
        input_text = DEFAULT_IMAGE_TOKEN + "\n" + input_text
    if video_path is not None:
        input_text = DEFAULT_VIDEO_TOKEN + "\n" + input_text
    if audio_path is not None:
        input_text = DEFAULT_AUDIO_TOKEN + "\n" + input_text

    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    print(f"Prompt: {full_prompt}")

    # Tokenize
    input_ids = tokenizer_multiple_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Process image if provided
    image_tensor = None
    if image_path is not None and image_processor is not None:
        image_data = load_image(image_path)
        image_tensor = (
            image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )

    # Set up signal token indices
    image_signal_token_indices = [
        tokenizer(f"<image_{i:02d}>").input_ids
        for i in range(model_config.n_img_tokens)
    ]
    video_signal_token_indices = [
        tokenizer(f"<video_{i:02d}>").input_ids
        for i in range(model_config.n_vid_tokens)
    ]
    audio_signal_token_indices = [
        tokenizer(f"<audio_{i:02d}>").input_ids
        for i in range(model_config.n_aud_tokens)
    ]

    # Generate response
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            images=image_tensor if image_tensor is not None else None,
            image_signal_token_indices=image_signal_token_indices,
            video_signal_token_indices=video_signal_token_indices,
            audio_signal_token_indices=audio_signal_token_indices,
            **generation_args.__dict__,
        )

    # Decode output
    output_text = tokenizer.batch_decode(
        output["sequences"], skip_special_tokens=True
    )[0]

    # Extract assistant response
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if stop_str in output_text:
        output_text = output_text[: output_text.index(stop_str)]

    # Clean up the response
    output_text = output_text.strip()

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="NExT-empthy: Multimodal Empathetic Response Generation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/empathy_finetune_1",
        help="Path to the fine-tuned empathy model",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Base model path (if using LoRA)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User input text",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (optional)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video (optional)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio (optional)",
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="empathy_v1",
        help="Conversation template to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit mode",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load model in 4-bit mode",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  NExT-empthy: Multimodal Empathy Model")
    print("=" * 50)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer, model, image_processor, video_processor, audio_processor, context_len, model_config = (
        setup_model(
            args.model_path,
            model_base=args.model_base,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
        )
    )
    print("Model loaded successfully.\n")

    # Show inputs
    print(f"User Input: {args.prompt}")
    if args.image:
        print(f"Image Input: {args.image}")
    if args.video:
        print(f"Video Input: {args.video}")
    if args.audio:
        print(f"Audio Input: {args.audio}")
    print()

    # Generate response
    response = generate_empathetic_response(
        tokenizer=tokenizer,
        model=model,
        model_config=model_config,
        image_processor=image_processor,
        prompt=args.prompt,
        image_path=args.image,
        video_path=args.video,
        audio_path=args.audio,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Empathetic Response:\n{response}")


if __name__ == "__main__":
    main()
