import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import numpy as np
import torch

logger = logging.getLogger(__name__)

URL_LINKS = {"dataset": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "asv"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LA.zip"
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in tqdm((self._data_dir / "LA").iterdir()):
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LA"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.9 * len(files))  # hand split, test ~ 10
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in tqdm(enumerate((self._data_dir / "wavs").iterdir())):
            if i < train_length:
                shutil.copy(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.copy(str(fpath), str(self._data_dir / "test" / fpath.name))
        # shutil.rmtree(str(self._data_dir / "wavs"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split("|")[0]
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists():  # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "audio_len": length,
                        }
                    )
        return index