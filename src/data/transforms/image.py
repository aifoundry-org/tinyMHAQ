import numpy as np

from PIL import Image
from typing import Tuple, Union
from enum import IntEnum


def pil_img_fromat(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img, mode="L")
    elif isinstance(img, Image.Image):
        pass
    else:
        raise TypeError(f"Wrong image type: {type(img)}!")

    return img


def image_resize(size: Tuple, interpolation: IntEnum = Image.Resampling.BILINEAR):
    def resize(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img = pil_img_fromat(img)

        img.resize(size=size, resample=interpolation)

        return np.asarray(img)

    return resize


def image_normalize(mean: np.ndarray, std: np.ndarray):
    def normalize(img: Union[np.ndarray]) -> np.ndarray:
        if img.max > 1:
            img /= 255

        return (img - mean) / std

    return normalize


def image_random_crop(
    crop_size: int,
    padding: Union[Tuple, list, int, None] = None,
    pad_if_needed=False,
    fill: int = 0,
    padding_mode: str = "constant",
):
    def random_crop(image: np.ndarray) -> np.ndarray:

        def apply_padding(img, pad_left, pad_top, pad_right, pad_bottom):
            if img.ndim == 3:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            elif img.ndim == 2:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            else:
                raise ValueError("Image must be 2D (grayscale) or 3D (color).")

            if padding_mode == "constant":
                return np.pad(img, pad_width, mode=padding_mode, constant_values=fill)
            else:
                return np.pad(img, pad_width, mode=padding_mode)

        if padding is not None:
            if isinstance(padding, int):
                pad_left = pad_top = pad_right = pad_bottom = padding
            elif isinstance(padding, (tuple, list)) and len(padding) == 2:
                pad_left = pad_right = padding[0]
                pad_top = pad_bottom = padding[1]
            elif isinstance(padding, (tuple, list)) and len(padding) == 4:
                pad_left, pad_top, pad_right, pad_bottom = padding
            else:
                raise ValueError(
                    "padding must be an int, a tuple of length 2, or a tuple of length 4."
                )
            image = apply_padding(image, pad_left, pad_top, pad_right, pad_bottom)

        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        if pad_if_needed:
            pad_bottom_needed = max(crop_h - h, 0)
            pad_right_needed = max(crop_w - w, 0)
            if pad_bottom_needed > 0 or pad_right_needed > 0:
                image = apply_padding(image, 0, 0, pad_right_needed, pad_bottom_needed)
                h, w = image.shape[:2]

        if h < crop_h or w < crop_w:
            raise ValueError("Image is smaller than the crop size after padding.")

        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

        if image.ndim == 3:
            return image[top : top + crop_h, left : left + crop_w, :]
        else:
            return image[top : top + crop_h, left : left + crop_w]

    return random_crop


def image_random_horizontal_flip(p: int = 0.5):
    def random_horizontal_flip(img: np.ndarray):
        if np.random.rand() < p:
            return img[:, ::-1, :]
        return img

    return random_horizontal_flip
