import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        config_parser: ConfigParser,
        limit=None,
        segment_size=None,
        max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.preprocessing = config_parser["preprocessing"]
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        self.segment_size = segment_size
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        if audio_wave.shape[-1] > self.segment_size:
            # start = np.random.randint(0, audio_wave.shape[-1] - self.segment_size + 1)
            # audio_wave = audio_wave[:, start : start + self.segment_size]
            audio_wave = audio_wave[:, : self.segment_size]
        elif audio_wave.shape[-1] < self.segment_size:
            while audio_wave.shape[-1] != self.segment_size:
                padding_value = min(
                    self.segment_size - audio_wave.shape[-1], audio_wave.shape[-1]
                )
                audio_wave = F.pad(audio_wave, (0, padding_value), mode="circular")
        return {
            "audio": audio_wave,
            "audio_type": 1 if data_dict["audio_type"] == "bonafide" else 0,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.preprocessing["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            audio_tensor_spec = self.wav2spec(audio_tensor_wave)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(index: list, max_audio_length, limit) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'audio_path'"
                " - duration of audio (in seconds)."
            )
