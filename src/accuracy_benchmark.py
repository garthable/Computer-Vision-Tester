from dataset import Dataset, Data
from typing import Any
import numpy as np
import cv2

def accuracy(data_predicted: Data, label_predicted: cv2.typing.Vec3i, 
             data_true: Data, label_true: cv2.typing.Vec3i) -> float:
    
    image_predicted: np.ndarray[int] = data_predicted.image.flatten()
    image_true: np.ndarray[int] = data_true.image.flatten()

    if len(image_predicted) != len(image_true):
        raise Exception(f'data predicted length ({len(data_predicted)}) is not equal to data truth length ({len(data_true)})')
    matching: int = 0
    for i in len(image_predicted):
        if image_predicted[i] == label_predicted:
            if image_true[i] == label_true:
                matching += 1
        else:
            if image_true[i] != label_true:
                matching += 1
    return float(matching) / float(len(image_predicted))

class AccuracyBenchmark:
    def __init__(self, ground_truth: Dataset, ground_truth_grass_label: cv2.typing.Vec3i):
        self.ground_truth: Dataset = ground_truth
        self.ground_truth_grass_label: cv2.typing.Vec3i = ground_truth_grass_label
    def get_accuracy(self, predicted: Dataset, predicted_grass_label: cv2.typing.Vec3i) -> float:
        results = []
        for i in range(len(predicted)):
            result: float = accuracy(predicted[i], predicted_grass_label, self.ground_truth, self.ground_truth_grass_label)
            results.append(result)
        return np.average(results)