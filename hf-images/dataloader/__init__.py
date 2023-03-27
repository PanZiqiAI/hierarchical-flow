
from torch.utils.data.dataloader import DataLoader
from custom_pkg.pytorch.operations import DataCycle
from image_datasets.datasets import CelebAHQ, FFHQ, AFHQ
from classification.datasets import get_dataset_without_labels_given_categories, ImageMNIST


def generate_data(cfg):
    # 1. Get dataset.
    if cfg.args.dataset == 'single-mnist':
        dataset = get_dataset_without_labels_given_categories(ImageMNIST, categories=[cfg.args.dataset_category], phase=cfg.args.dataset_phase)
    elif cfg.args.dataset == 'celeba-hq':
        dataset = CelebAHQ(image_size=cfg.args.img_size, max_dataset_size=cfg.args.dataset_maxsize)
    elif cfg.args.dataset == 'ffhq':
        dataset = FFHQ(image_size=cfg.args.img_size, max_dataset_size=cfg.args.dataset_maxsize)
    elif cfg.args.dataset == 'afhq':
        dataset = AFHQ(image_size=cfg.args.img_size, category=cfg.args.dataset_category, phase=cfg.args.dataset_phase, max_dataset_size=cfg.args.dataset_maxsize)
    else: raise NotImplementedError
    # 2. Get dataloader.
    dataloader = DataLoader(
        dataset, batch_size=cfg.args.batch_size, drop_last=cfg.args.dataset_drop_last,
        shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)
    # Return
    return DataCycle(dataloader)
