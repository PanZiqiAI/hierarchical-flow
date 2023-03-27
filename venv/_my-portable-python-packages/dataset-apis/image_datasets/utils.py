
import os


########################################################################################################################
# Image folder
########################################################################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root_dir, max_dataset_size=float("inf")):
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
    # 1. Init
    images = []
    # 2. Prepare
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
