
import os
from PIL import Image
from .utils import make_dataset
from torch.utils.data import Dataset
from __data_root__ import __DATA_ROOT__
from torchvision.datasets.folder import default_loader
from torchvision import transforms as torch_transforms


def default_transforms(**kwargs):
    # 1. Init result
    transform_list = []
    # 2. Transforms
    # (1) To grayscale
    try:
        if kwargs['grayscale']: transform_list.append(torch_transforms.Grayscale(1))
    except KeyError:
        pass
    # (2) Resize or scale
    if 'load_size' in kwargs.keys():
        transform_list.append(torch_transforms.Resize(kwargs['load_size'], Image.BICUBIC))
    # (3) Crop
    if 'crop_size' in kwargs.keys():
        # Get crop type
        crop_func = torch_transforms.RandomCrop
        if 'crop_type' in kwargs.keys():
            crop_func = {
                'random': torch_transforms.RandomCrop,
                'center': torch_transforms.CenterCrop
            }[kwargs['crop_type']]
        # Save
        transform_list.append(crop_func(kwargs['crop_size']))
    # (4) Flip
    try:
        if kwargs['flip']: transform_list.append(torch_transforms.RandomHorizontalFlip())
    except KeyError:
        pass
    # (5) To tensor
    try:
        if kwargs['to_tensor']: transform_list += [torch_transforms.ToTensor()]
    except KeyError:
        pass
    # (6) Normalize
    if 'normalize' in kwargs.keys():
        # Get normalize
        if kwargs['normalize'] == 'default':
            if 'grayscale' in kwargs.keys() and kwargs['grayscale']: norm = torch_transforms.Normalize((0.5, ), (0.5, ))
            else: norm = torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm = torch_transforms.Normalize(*kwargs['normalize'])
        # Save
        transform_list.append(norm)
    # Return
    return torch_transforms.Compose(transform_list)


def default_paired_transforms(grayscales, **kwargs):
    return [
        # Transforms a
        default_transforms(grayscale=grayscales[0], **kwargs),
        # Transforms b
        default_transforms(grayscale=grayscales[1], **kwargs)
    ]


########################################################################################################################
# Aligned Datasets
########################################################################################################################

class AlignedImageDataset(Dataset):
    """
        A dataset class for paired image dataset.

        It assumes that the root_dir contains image pairs in the form of {A,B}.
    """
    def __init__(self, root_dir, direction, transforms, max_dataset_size=float("inf")):
        super(AlignedImageDataset, self).__init__()
        # Config
        self._ab_paths = sorted(make_dataset(root_dir, max_dataset_size))
        self._direction = direction
        # Transforms
        self._transforms_a, self._transforms_b = transforms

    def __getitem__(self, index):
        """
        :return: A (tensor), B (tensor)
        """
        # 1. Load & split
        # Load pair
        ab = default_loader(self._ab_paths[index])
        # Split images
        w, h = ab.size
        w2 = int(w / 2)
        a = ab.crop((0, 0, w2, h))
        b = ab.crop((w2, 0, w, h))
        # 2. Transform
        a = self._transforms_a(a)
        b = self._transforms_b(b)
        # Return
        if self._direction == 'ab':
            return {'a': a, 'b': b}
        elif self._direction == 'ba':
            return {'a': b, 'b': a}
        else:
            raise ValueError

    def __len__(self):
        return len(self._ab_paths)


# ----------------------------------------------------------------------------------------------------------------------
# edges2shoes
# ----------------------------------------------------------------------------------------------------------------------

class Edge2Shoes(AlignedImageDataset):
    """
    edges2shoes dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'val']
        # Init
        super(Edge2Shoes, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, "edges2shoes/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([True, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# edges2handbags
# ----------------------------------------------------------------------------------------------------------------------

class Edge2Handbags(AlignedImageDataset):
    """
    edges2shoes dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'val']
        # Init
        super(Edge2Handbags, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__,  "edges2handbags/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([True, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# night2day
# ----------------------------------------------------------------------------------------------------------------------

class Night2Day(AlignedImageDataset):
    """
    night2day dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'test', 'val']
        # Init
        super(Night2Day, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__,  "night2day/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([False, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# facades
# ----------------------------------------------------------------------------------------------------------------------

class Facades(AlignedImageDataset):
    """
    Facades dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'test', 'val']
        # Init
        super(Facades, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__,  "facades/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([False, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# cityscapes
# ----------------------------------------------------------------------------------------------------------------------

class Cityscapes(AlignedImageDataset):
    """
    Facades dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'val']
        # Init
        super(Cityscapes, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__,  "cityscapes/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([False, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------------------------------

class Maps(AlignedImageDataset):
    """
    Facades dataset.
    """
    def __init__(self, phase, direction, max_dataset_size=float("inf"), **kwargs):
        assert phase in ['train', 'val']
        # Init
        super(Maps, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__,  "maps/%s" % phase), max_dataset_size=max_dataset_size, direction=direction,
            transforms=default_paired_transforms([False, False], to_tensor=True, normalize='default', **kwargs) 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


########################################################################################################################
# Single Datasets
########################################################################################################################

class SingleImageDataset(Dataset):
    """
        A dataset class for single image dataset.
        It assumes that the root_dir contains images.
    """
    def __init__(self, root_dir, transforms, max_dataset_size=float("inf")):
        super(SingleImageDataset, self).__init__()
        # Path
        self._paths = sorted(make_dataset(root_dir, max_dataset_size))
        # Transforms
        self._transforms = transforms

    def __getitem__(self, index):
        """
        :return: An image.
        """
        # 1. Load
        img = default_loader(self._paths[index])
        # 2. Transform
        img = self._transforms(img)
        # Return
        return img

    def __len__(self):
        return len(self._paths)


# ----------------------------------------------------------------------------------------------------------------------
# celeba
# ----------------------------------------------------------------------------------------------------------------------

class CelebAHQ(SingleImageDataset):
    """
    CelebA-HQ dataset.
    """
    def __init__(self, image_size, max_dataset_size=float("inf"), **kwargs):
        super(CelebAHQ, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, 'celeba-hq/celeba-%d' % image_size), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default') 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


class CelebAAlign(SingleImageDataset):
    """
    CelebA-Align dataset.
    """
    def __init__(self, max_dataset_size=float("inf"), **kwargs):
        super(CelebAAlign, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, 'celeba/Img/img_align_celeba'), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default') 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# AFHQ
# ----------------------------------------------------------------------------------------------------------------------

class AFHQ(SingleImageDataset):
    """
    AFHQ dataset.
    """
    def __init__(self, image_size, category, phase='train', max_dataset_size=float("inf"), **kwargs):
        super(AFHQ, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, 'afhq/images%dx%d' % (image_size, image_size), phase, category), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# FFHQ
# ----------------------------------------------------------------------------------------------------------------------

class FFHQ(SingleImageDataset):
    """
    FFHQ dataset.
    """
    def __init__(self, image_size, max_dataset_size=float("inf"), **kwargs):
        super(FFHQ, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, 'ffhq/images%dx%d' % (image_size, image_size)), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# LSUN
# ----------------------------------------------------------------------------------------------------------------------

class LSUN(SingleImageDataset):
    """
    LSUN dataset.
    """
    def __init__(self, selection, image_size=None, max_dataset_size=float("inf"), **kwargs):
        super(LSUN, self).__init__(
            root_dir=os.path.join(__DATA_ROOT__, 'lsun', 'images' if image_size is None else 'images%dx%d' % (image_size, image_size), selection),
            max_dataset_size=max_dataset_size, transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])
