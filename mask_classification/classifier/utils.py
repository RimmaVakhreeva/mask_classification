from pathlib import Path

import torch
import tqdm
from pytorch_lightning import seed_everything

__all__ = ['load_images', 'seed_everything_deterministic']


def load_images(folder, masks_folder_name='with_mask'):
    output_data = []
    img_suffixes = {'.png', '.jpg', '.jpeg'}
    for image_filename in tqdm.tqdm(folder.iterdir()):
        if image_filename.suffix.lower() not in img_suffixes:
            continue
        output_data.append((image_filename, int(image_filename.parent.name == masks_folder_name)))
    return output_data


def load_module(module_filename: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(__name__, str(module_filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
