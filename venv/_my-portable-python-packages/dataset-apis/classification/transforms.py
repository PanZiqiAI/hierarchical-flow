
import numpy as np


class Flatten(object):
    """
    To vector.
    """
    def __call__(self, image):
        return image.view(-1, )


class To32x32(object):
    """
    From 28x28 -> 32x32.
    """
    def __call__(self, x):
        assert x.shape == (28, 28, 1)
        x = np.concatenate([np.zeros(shape=(28, 2, 1), dtype=x.dtype), x,
                            np.zeros(shape=(28, 2, 1), dtype=x.dtype)], axis=1)
        x = np.concatenate([np.zeros(shape=(2, 32, 1), dtype=x.dtype), x,
                            np.zeros(shape=(2, 32, 1), dtype=x.dtype)], axis=0)
        return x


class RandomColor(object):
    """
    Given an image of np.array (H, W, 1), randomly assign a color. The output ranges in (-1, 1).
    """
    def __call__(self, x):
        # Sample a color & apply.
        color = np.random.normal(loc=0.0, scale=0.5, size=(1, 1, 3)).astype("float32")
        x = x*color + (x-1.0)
        # Add noise.
        noise = np.random.normal(loc=0.0, scale=0.2, size=x.shape).astype("float32")
        x = x + noise
        # Return
        return np.clip(x, -1.0, 1.0)
