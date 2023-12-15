import logging
from pathlib import Path

from src.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, segment_size=64000, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, segment_size * args, **kwargs)
