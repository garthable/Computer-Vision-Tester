from time_benchmark import TimeBenchmark
from dataset import Dataset, Data, Image

GROUND_TRUTH_LABEL = [0, 0, 0]
PREDICTED_LABEL = [0, 0, 0]
NAME = ""

def run_segmentation(input_image: Image) -> Image:
    return input_image

def benchmark(data_path: str) -> None:
    data = Dataset()
    data.load_from_path(data_path)
    predicted = Dataset()
    
    timer = TimeBenchmark()

    for i in range(len(data)):
        timer.begin_benchmark()
        
        image: Image = run_segmentation(data[i].image)

        timer.end_benchmark()
        predicted.append(Data(image=image))

    print(f'Time per frame: {timer.get_average_time()}')
    
