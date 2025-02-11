from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from dataset import get_leaf_files
import numpy as np
import cv2

label_to_color = {}

def save_ground_truth(result_dict, path: str):
    np.random.seed(0)
    COLORS = np.array(np.random.choice(range(256), size=3*256)).reshape(256, 3)
    
    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
    img = None
    # Plot for each segment area
    for i, val in enumerate(sorted_result):
        print(val)
        mask = val['segmentation']
        mask_image = mask.astype(np.uint8)
        color_image = np.zeros(shape=(mask_image.shape[0],mask_image.shape[1],3))
        for y in range(len(mask_image)):
            for x in range(len(mask_image[y])):
                color = COLORS[i]*mask_image[y, x]
                color_image[y, x] = color
        if img is None:
            img = color_image
        else:
            img += color_image
    path = path.replace('data', 'ground_truth')
    print(path)
    cv2.imwrite(path, img)

def generate_ground_truth(image: np.ndarray[int], path: str) -> None:
    sam = sam_model_registry["default"](checkpoint="model/sam_vit_h_4b8939.pth")
    print('make sam')
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("make mask generator")
    result = mask_generator.generate(image)
    print("masks generated")
    save_ground_truth(result, path)

if __name__ == '__main__':
    files = get_leaf_files('data')
    for file in files:
        print(file)
        image = cv2.imread(file)
        generate_ground_truth(image, file)