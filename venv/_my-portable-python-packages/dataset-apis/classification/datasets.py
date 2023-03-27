
import os
import sys
import copy
import torch
import pickle
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from __data_root__ import __DATA_ROOT__
from torchvision.utils import save_image
from torchvision import transforms as torch_t
from torchvision.datasets.folder import default_loader
from .transforms import Flatten, To32x32, RandomColor
from .utils import decode_idx1_ubyte, decode_idx3_ubyte, make_dataset


########################################################################################################################
# Dataset
########################################################################################################################

class BaseClassification(Dataset):
    """
    Base class for dataset for classification.
    """
    _data = None
    _label: np.ndarray      # Each element is a integer indicating the category index.

    def __init__(self):
        # 1. Samples per category
        self._sample_indices = [np.argwhere(self._label == y)[:, 0].tolist() for y in range(self.num_classes)]
        # 2. Class counter
        self._class_counter = [len(samples) for samples in self._sample_indices]

    @property
    def num_classes(self):
        """
        :return: A integer indicating number of categories.
        """
        return len(set(self._label))

    @property
    def class_counter(self):
        """
        :return: A list, whose i-th element equals to the total sample number of the i-th category.
        """
        return self._class_counter

    @property
    def sample_indices(self):
        """
        :return: A list, whose i-th element is a numpy.array containing sample indices of the i-th category.
        """
        return self._sample_indices

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        """
        Should return x, y, where y is the class label.
        :param index:
        :return:
        """
        raise NotImplementedError

    def subset(self, categories):
        # 1. Create an instance
        dataset = copy.deepcopy(self)
        class_idx_to_orig = {}
        # 2. Modify
        dataset._data, dataset._label = [], []
        for cat_index, y in enumerate(categories):
            # Data & label
            for index in self._sample_indices[y]:
                dataset._data.append(self._data[index])
                dataset._label.append(cat_index)
            # Idx
            class_idx_to_orig[cat_index] = y
        # (3) Get label
        dataset._label = np.array(dataset._label, dtype=np.int64)
        # Initialize
        BaseClassification.__init__(dataset)
        setattr(dataset, 'class_idx_to_orig', class_idx_to_orig)
        # Return
        return dataset


########################################################################################################################
# Instances
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MNIST & FashionMNIST
# ----------------------------------------------------------------------------------------------------------------------

def mnist_paths(name):
    assert name in ['mnist', 'fashion_mnist']
    return {
        'train': (os.path.join(__DATA_ROOT__, "%s/train-images.idx3-ubyte" % name),
                  os.path.join(__DATA_ROOT__, "%s/train-labels.idx1-ubyte" % name)),
        'test': (os.path.join(__DATA_ROOT__, "%s/t10k-images.idx3-ubyte" % name),
                 os.path.join(__DATA_ROOT__, "%s/t10k-labels.idx1-ubyte" % name))}


class MNIST(BaseClassification):
    """
    MNIST dataset.
    """
    def __init__(self, images_path, labels_path, transforms=None):
        # Data & label
        self._data = decode_idx3_ubyte(images_path).astype('uint8')[:, :, :, np.newaxis]
        self._label = decode_idx1_ubyte(labels_path).astype('int64')
        # Transforms
        self._transforms = transforms
        # Initialize
        super(MNIST, self).__init__()

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._data[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label


class FlattenMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        transforms = torch_t.Compose([torch_t.ToTensor(), torch_t.Normalize((0.5, ), (0.5, )), Flatten()]) \
            if 'transforms' not in kwargs.keys() else kwargs['transforms']
        # Init
        super(FlattenMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


class ImageMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = kwargs['transforms']
        else:
            transforms = [torch_t.ToTensor(), torch_t.Normalize((0.5, ), (0.5, ))]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [To32x32()] + transforms
            transforms = torch_t.Compose(transforms)
        # Init
        super(ImageMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


def get_dataset_without_labels_given_categories(base_class, categories, *args, **kwargs):
    """
    :param base_class:
    :param categories: List of integers.
    :param args:
    :param kwargs:
    :return:
    """

    class _DatasetWithoutLabels(base_class):
        """
        Reimplement __getitem__.
        """
        def __getitem__(self, item):
            _x, _y = super(_DatasetWithoutLabels, self).__getitem__(item)
            return _x

    return _DatasetWithoutLabels(*args, **kwargs).subset(categories=categories)


# ----------------------------------------------------------------------------------------------------------------------
# Colored MNIST
# ----------------------------------------------------------------------------------------------------------------------

class ColoredMNIST(MNIST):
    """
    Colored MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = torch_t.Compose([RandomColor(), kwargs['transforms']])
        else:
            transforms = [RandomColor(), torch_t.ToTensor()]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [To32x32()] + transforms
            transforms = torch_t.Compose(transforms)
        # Init
        super(ColoredMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


# ----------------------------------------------------------------------------------------------------------------------
# CIFAR-10
# ----------------------------------------------------------------------------------------------------------------------

class CIFAR10(BaseClassification):
    """
    CIFAR-10 for classification.
    """
    def __init__(self, phase, transforms=torch_t.Compose([torch_t.ToTensor(), torch_t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        # --------------------------------------------------------------------------------------------------------------
        # Data & label
        # --------------------------------------------------------------------------------------------------------------
        assert phase in ['train', 'test']
        # 1. Init results.
        data, labels = [], []
        # 2. Collect from each batch
        batch_filenames = ['data_batch_%d' % i for i in range(1, 6)] if phase == 'train' else ['test_batch']
        for filename in batch_filenames:
            entry = self._load_file(os.path.join(filename))
            """ Saving """
            data.append(entry['data'])
            labels.extend(entry['labels'])
        # 3. Set data & label
        data = np.vstack(data).reshape(-1, 3, 32, 32)
        self._data, self._label = data.swapaxes(1, 2).swapaxes(2, 3), np.array(labels, dtype=np.int64)
        """ Set class names. """
        self._class_names = self._load_file(os.path.join('batches.meta'))['label_names']
        # --------------------------------------------------------------------------------------------------------------
        # Transforms
        # --------------------------------------------------------------------------------------------------------------
        self._transforms = transforms
        """ Initialize """
        super(CIFAR10, self).__init__()

    @property
    def class_names(self):
        return self._class_names

    @staticmethod
    def _load_file(filename):
        with open(os.path.join(__DATA_ROOT__, 'cifar10', 'cifar-10-batches-py', filename), 'rb') as f:
            if sys.version_info[0] == 2: return pickle.load(f)
            else: return pickle.load(f, encoding='latin1')

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._data[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label


# ----------------------------------------------------------------------------------------------------------------------
# CelebA-HQ based on binary attributes
# ----------------------------------------------------------------------------------------------------------------------

def default_transforms(**kwargs):
    # 1. Init result
    transform_list = []
    # 2. Transforms
    # (1) To grayscale
    try:
        if kwargs['grayscale']: transform_list.append(torch_t.Grayscale(1))
    except KeyError:
        pass
    # (2) Resize or scale
    if 'load_size' in kwargs.keys():
        transform_list.append(torch_t.Resize(kwargs['load_size'], Image.BICUBIC))
    # (3) Crop
    if 'crop_size' in kwargs.keys():
        # Get crop type
        crop_func = torch_t.RandomCrop
        if 'crop_type' in kwargs.keys():
            crop_func = {
                'random': torch_t.RandomCrop,
                'center': torch_t.CenterCrop
            }[kwargs['crop_type']]
        # Save
        transform_list.append(crop_func(kwargs['crop_size']))
    # (4) Flip
    try:
        if kwargs['flip']: transform_list.append(torch_t.RandomHorizontalFlip())
    except KeyError:
        pass
    # (5) To tensor
    try:
        if kwargs['to_tensor']: transform_list += [torch_t.ToTensor()]
    except KeyError:
        pass
    # (6) Normalize
    if 'normalize' in kwargs.keys():
        # Get normalize
        if kwargs['normalize'] == 'default':
            if 'grayscale' in kwargs.keys() and kwargs['grayscale']: norm = torch_t.Normalize((0.5, ), (0.5, ))
            else: norm = torch_t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm = torch_t.Normalize(*kwargs['normalize'])
        # Save
        transform_list.append(norm)
    # Return
    return torch_t.Compose(transform_list)


class CelebAHQ(BaseClassification):
    """ CelebA-HQ classification dataset based on binary attributes. All attributes:
        5_o_Clock_Shadow
        Arched_Eyebrows
        Attractive
        Bags_Under_Eyes
        Bald
        Bangs
        Big_Lips
        Big_Nose
        Black_Hair
        Blond_Hair
        Blurry
        Brown_Hair
        Bushy_Eyebrows
        Chubby
        Double_Chin
        Eyeglasses
        Goatee
        Gray_Hair
        Heavy_Makeup
        High_Cheekbones
        Male
        Mouth_Slightly_Open
        Mustache
        Narrow_Eyes
        No_Beard
        Oval_Face
        Pale_Skin
        Pointy_Nose
        Receding_Hairline
        Rosy_Cheeks
        Sideburns
        Smiling
        Straight_Hair
        Wavy_Hair
        Wearing_Earrings
        Wearing_Hat
        Wearing_Lipstick
        Wearing_Necklace
        Wearing_Necktie
        Young
    """
    def __init__(self, attributes, image_size, max_dataset_size=float("inf"), **kwargs):
        # Config.
        self._attributes = attributes
        # Data & label
        self._data = sorted(make_dataset(
            root_dir=os.path.join(__DATA_ROOT__, 'celeba-hq/celeba-%d' % image_size), max_dataset_size=max_dataset_size))
        self._label = self._generate_label()
        # Transforms
        self._transforms = default_transforms(**kwargs, to_tensor=True, normalize='default') if 'transforms' not in kwargs else kwargs['transforms']
        # Initialize
        super(CelebAHQ, self).__init__()

    @property
    def attributes(self):
        return self._attributes

    def _generate_label(self):
        # Get annotations.
        attr_annos, all_attrs = [], None
        with open(os.path.join(__DATA_ROOT__, 'celeba/Anno/list_attr_celeba.txt'), mode='r') as f:
            # Get all attributes & check.
            f.readline()
            all_attrs = f.readline().split()
            for _a in self._attributes: assert _a in all_attrs, "Unrecognized attribute '%s'. " % _a
            # Get attribute values for every image.
            cursor_data = 0
            while True:
                line = f.readline().split()
                if not line: break
                # Read line.
                filename, attr_values = line[0], line[1:]
                if os.path.split(self._data[cursor_data])[1] != filename: continue
                """ Saving. """
                attr_annos.append(np.array([int(_a) > 0 for _a in attr_values])[None])
                cursor_data += 1
                if cursor_data == len(self._data): break
        attr_annos = np.concatenate(attr_annos)
        # To category label.
        attr_annos = np.concatenate([attr_annos[:, all_attrs.index(_a)][:, None] for _a in self._attributes], axis=1).astype(dtype=np.int64)
        bases = np.array([2**_i for _i in range(len(self._attributes))][::-1]).astype(dtype=np.int64)
        label = (attr_annos*bases[None]).sum(1).astype(dtype=np.int64)
        # Return
        return label

    def __getitem__(self, index):
        # 1. Load image & transform.
        img = default_loader(self._data[index])
        img = self._transforms(img)
        # 2. Load label.
        label = self._label[index]
        # Return
        return img, label

    def __len__(self):
        return len(self._data)

    def visualize(self, vis_dir, num_per_cat=64):
        """ Visualize each category. """
        for category in range(self.num_classes):
            # Get visualizing path for current category.
            attr_values = [_s for _s in bin(category)[2:]]
            if len(attr_values) < len(self._attributes): attr_values = ['0']*(len(self._attributes)-len(attr_values)) + attr_values
            cat_vis_file = "%d-" % category
            for attr, value in zip(self._attributes, attr_values): cat_vis_file += "%s@%s-" % (attr, value)
            cat_vis_file = cat_vis_file[:-1] + ".png"
            # Random sampling.
            rnd_indices = random.sample(self.sample_indices[category], k=num_per_cat)
            images = np.concatenate([np.array(default_loader(self._data[_i]))[None] for _i in rnd_indices]).swapaxes(2, 3).swapaxes(1, 2) / 255.0
            """ Visualize. """
            save_image(torch.from_numpy(images), os.path.join(vis_dir, cat_vis_file))
