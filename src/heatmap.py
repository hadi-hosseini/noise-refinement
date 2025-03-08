import re, os, shutil
from compel import Compel
import difflib
import numpy as np
from PIL import Image
from typing import Union, Dict
import cv2

import torch
import torch.nn.functional as F

from src.vis_utils import show_image_relevance
from src.ptp_utils import text_under_image


def normalize_indices_to_alter(indices_to_alter):
    indices_to_alter = [index - 1 for index in indices_to_alter]
    return indices_to_alter

def show_cross_attention(prompt: str, attention_maps: np.ndarray, tokenizer, indices_to_alter, res: int, orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    indices_to_alter = normalize_indices_to_alter(indices_to_alter)
    images = []

    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = text_under_image(image, decoder(int(tokens[i + 1])))
            images.append(image)
    
    return images


def view_images(images, num_rows: int = 1, offset_ratio: float = 0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[i * num_cols + j]
    return image_  

def create_full_heatmap_image(heatmap_data):
    heatmap_images = [text_under_image(entity['heatmap_image'], f"step {entity['step']}") for entity in heatmap_data]
    heatmap_image = view_images(heatmap_images, num_rows=10, offset_ratio=0.1)
    return heatmap_image


def create_latent_heatmap_image(latents, attention_maps, prompt, indices_to_alter, pipe) -> np.ndarray:
    decoded_image = pipe.numpy_to_pil(pipe.decode_latents(latents))[0]
    heatmap_image = show_cross_attention(attention_maps=attention_maps,
                                   prompt=prompt,
                                   tokenizer=pipe.tokenizer,
                                   res=16,
                                   indices_to_alter=indices_to_alter,
                                   orig_image=decoded_image)
    heatmap_image = view_images(heatmap_image, num_rows=1)
    return heatmap_image


def convert_numpy_to_pil(image: np.ndarray):
    return Image.fromarray(image)

