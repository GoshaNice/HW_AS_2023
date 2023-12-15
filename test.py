import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_arch
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from src.metric.compute_eer import compute_eer
import torchaudio
import torch.nn.functional as F
import numpy as np

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, input_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    for path in sorted(Path(input_dir).iterdir()):
        entry = {}
        if path.suffix not in [".mp3", ".wav", ".flac", ".m4a"]:
            continue
        
        audio_wave, sr = torchaudio.load(path)
        if sr != 16000:
            audio_wave = torchaudio.functional.resample(audio_wave, sr, 16000)
        if audio_wave.shape[-1] > 64000:
            audio_wave = audio_wave[:, : 64000]
        elif audio_wave.shape[-1] < 64000:
            while audio_wave.shape[-1] != 64000:
                padding_value = min(
                    64000 - audio_wave.shape[-1], audio_wave.shape[-1]
                )
                audio_wave = F.pad(audio_wave, (0, padding_value), mode="circular")
        
        with torch.no_grad():
            audio_wave = audio_wave.to(device)
            output = model(audio = audio_wave)
            prediction = output["prediction"]
        
        print(f"For audio {path} probability bonafied is {prediction[0][1]}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default="test_data_folder",
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=5,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))


    main(config, args.test_data_folder)