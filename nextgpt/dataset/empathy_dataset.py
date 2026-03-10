"""Dataset for multimodal empathy fine-tuning.

This dataset extends the base LazySupervisedDataset to handle empathy-specific
data with emotion labels. It supports multimodal inputs (text, image, video, audio)
and produces empathetic text responses with optional emotion classification.

Expected data JSON format:
[
    {
        "emotion": "sad",
        "conversations": [
            {"from": "human", "value": "I just lost my pet..."},
            {"from": "gpt", "value": "I'm so sorry to hear that..."}
        ],
        "input_image": "image_file.jpg",   // optional
        "input_video": "video_file.mp4",   // optional
        "input_audio": "audio_file.wav"    // optional
    },
    ...
]
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from nextgpt.constants import (
    EMOTION_LABELS,
    EMOTION_TO_IDX,
    IGNORE_INDEX,
    NUM_EMOTION_LABELS,
)

from .base_dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset


class EmpathyDataset(LazySupervisedDataset):
    """Dataset for multimodal empathy fine-tuning.

    Extends LazySupervisedDataset to add emotion label handling.
    Each sample can optionally include an 'emotion' field that is
    converted to a classification label.
    """

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = super().__getitem__(i)

        # Add emotion label if present in the data
        sources = self.list_data_dict[i]
        if "emotion" in sources:
            emotion = sources["emotion"]
            if emotion in EMOTION_TO_IDX:
                data_dict["emotion_label"] = torch.tensor(
                    EMOTION_TO_IDX[emotion], dtype=torch.long
                )
            else:
                # Unknown emotion label defaults to -1 (ignored in loss)
                data_dict["emotion_label"] = torch.tensor(-1, dtype=torch.long)
        else:
            data_dict["emotion_label"] = torch.tensor(-1, dtype=torch.long)

        return data_dict


@dataclass
class EmpathyDataCollator(DataCollatorForSupervisedDataset):
    """Collate examples for empathy fine-tuning.

    Extends base collator with emotion label batching.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)

        # Collate emotion labels
        emotion_labels = []
        for instance in instances:
            if "emotion_label" in instance:
                emotion_labels.append(instance["emotion_label"])

        if len(emotion_labels) > 0:
            batch["emotion_labels"] = torch.stack(emotion_labels, dim=0)

        return batch
