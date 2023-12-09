import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    audios = []
    result_batch["audio_length"] = torch.tensor(
        [item["audio"].shape[-1] for item in dataset_items]
    )
    max_audio_dim_last = torch.max(result_batch["audio_length"])
    for item in dataset_items:
        audio = item["audio"]
        audios.append(
            F.pad(
                audio,
                (0, max_audio_dim_last - audio.shape[-1]),
                "constant",
                0,
            )
        )

    result_batch["audio"] = torch.cat(audios, dim=0)
    result_batch["target"] = torch.tensor(
        [item["audio_type"] for item in dataset_items]
    )
    return result_batch
