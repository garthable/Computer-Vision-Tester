import numpy as np
import cv2
import os.path as path 
import os

from typing import Any
from preprocess import preprocess

Image = cv2.typing.MatLike

class Data:
    def __init__(self, path: str | None = None, image: Image | None = None):
        self.path: str | None = path
        self.image: Image | None = None
        if path is not None:
            self.image = cv2.imread(path)
        elif image is not None:
            self.image = image
    def __getitem__(self, i: int):
        return self.image[i]
    def __len__(self):
        return len(self.image)

def get_leaf_files(dir_path) -> list[str]:
    files: list[str] = []
    for root, dirs, file_names in os.walk(dir_path):
        if not dirs:
            for file_name in file_names:
                files.append(path.join(root, file_name))
    return files

class Dataset:
    def __init__(self):
        self.data: list[Data] = []
    def load_from_path(self, parent_path: str) -> None:
        paths: list[str] = get_leaf_files(parent_path)
        for path in paths:
            self.data.append(Data(path=path))
    def load_from_images(self, images: list[Image]) -> None:
        for image in images:
            self.data.append(Data(image=image))
    def preprocess(self) -> None:
        for i in range(len(self.data)):
            self.data[i] = preprocess(self.data[i])
    def append(self, data: Data) -> None:
        self.data.append(data)
    def __getitem__(self, i: int) -> Data:
        return self.data[i]
    def __len__(self):
        return len(self.data)