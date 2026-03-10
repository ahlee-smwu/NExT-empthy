# Empathy Dataset Preparation

This directory contains data for training the multimodal empathy model.

## Directory Structure

```
data/empathy/
├── empathy_text_instruction.json         # Text-only empathy conversations
├── empathy_multimodal_instruction.json   # Multimodal empathy conversations
├── images/                               # Image files for visual emotion inputs
├── videos/                               # Video files for visual+temporal emotion inputs
├── audios/                               # Audio files for vocal emotion inputs
└── README.md                             # This file
```

## Data Format

### Text-Only Empathy Data (`empathy_text_instruction.json`)

Each sample follows this format:
```json
{
    "emotion": "sad",
    "conversations": [
        {"from": "human", "value": "I just lost my pet..."},
        {"from": "gpt", "value": "I'm so sorry to hear that..."}
    ]
}
```

### Multimodal Empathy Data (`empathy_multimodal_instruction.json`)

Each sample can include optional multimodal inputs:
```json
{
    "emotion": "afraid",
    "input_image": "scared_face.jpg",
    "input_video": "emotional_moment.mp4",
    "input_audio": "crying_sound.wav",
    "conversations": [
        {"from": "human", "value": "<image>\nI feel terrible today."},
        {"from": "gpt", "value": "I can see you're going through..."}
    ]
}
```

**Notes:**
- Use `<image>`, `<video>`, or `<audio>` tokens in the conversation `value` field to indicate where multimodal inputs should be placed.
- The `emotion` field should be one of the 32 supported emotion labels (see below).
- Each conversation should have alternating `human` and `gpt` messages.
- The `gpt` messages should demonstrate empathetic understanding and emotionally appropriate responses.

## Supported Emotion Labels

The model supports the following 32 emotion labels:

| Emotion | Emotion | Emotion | Emotion |
|---------|---------|---------|---------|
| surprised | excited | angry | proud |
| sad | annoyed | grateful | lonely |
| afraid | terrified | guilty | impressed |
| disgusted | hopeful | confident | furious |
| anxious | anticipating | joyful | nostalgic |
| disappointed | prepared | jealous | content |
| devastated | sentimental | embarrassed | caring |
| trusting | ashamed | apprehensive | faithful |

## Preparing Your Own Empathy Dataset

### Recommended Public Datasets

You can use the following public datasets to create empathy training data:

1. **EmpatheticDialogues** (Facebook Research)
   - ~25K conversations grounded in emotional situations
   - Contains emotion labels for each conversation
   - Download: https://github.com/facebookresearch/EmpatheticDialogues

2. **IEMOCAP** (USC)
   - Multimodal dataset with audio, video, and text
   - Emotion labels: happy, sad, angry, neutral, frustrated, excited
   - Suitable for multimodal emotion recognition

3. **MELD** (Multimodal EmotionLines Dataset)
   - Multimodal multi-party conversations from TV series
   - Includes audio, video, and text with emotion annotations
   - Download: https://github.com/declare-lab/MELD

4. **CMU-MOSEI** (CMU Multimodal Opinion Sentiment and Emotion Intensity)
   - Video segments with sentiment and emotion annotations
   - Covers 6 basic emotions with intensity scores

### Conversion Script

Use `data/empathy/prepare_empathy_data.py` to convert these datasets into the required format.
