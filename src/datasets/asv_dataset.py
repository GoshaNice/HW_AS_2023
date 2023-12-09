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


class ASVDataset(BaseDataset):
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

        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "dev").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "eval").mkdir(exist_ok=True, parents=True)

        for i, fpath in tqdm(enumerate((self._data_dir / "LA" / "ASVspoof2019_LA_train" / "flac").iterdir())):
            shutil.copy(str(fpath), str(self._data_dir / "train" / fpath.name))
        
        for i, fpath in tqdm(enumerate((self._data_dir / "LA" / "ASVspoof2019_LA_dev" / "flac").iterdir())):
            shutil.copy(str(fpath), str(self._data_dir / "dev" / fpath.name))
            
        for i, fpath in tqdm(enumerate((self._data_dir / "LA" / "ASVspoof2019_LA_eval" / "flac").iterdir())):
            shutil.copy(str(fpath), str(self._data_dir / "eval" / fpath.name))
        
        shutil.copy(str(self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"), str(self._data_dir / "meta_train.txt"))
        shutil.copy(str(self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"), str(self._data_dir / "meta_eval.txt"))
        shutil.copy(str(self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"), str(self._data_dir / "meta_dev.txt"))

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

        meta_file = self._data_dir / f"meta_{part}.txt"
        with open(str(meta_file), "r") as f:
            data = f.readlines()
        for i in tqdm(range(len(data))):
            row = data[i].strip().split()
            name = row[1]
            audio_type = row[-1]
            wav_path = split_dir / f"{name}.flac"
            t_info = torchaudio.info(str(wav_path))
            length = t_info.num_frames / t_info.sample_rate

            index.append(
                {
                    "path": str(wav_path),
                    "audio_type": audio_type,
                    "audio_len": length,
                }
            )
        
        return index
