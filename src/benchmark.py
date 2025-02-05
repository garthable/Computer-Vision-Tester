from time_benchmark import TimeBenchmark
from accuracy_benchmark import AccuracyBenchmark
from dataset import Dataset, Data, Image

GROUND_TRUTH_LABEL = [0, 0, 0]
PREDICTED_LABEL = [0, 0, 0]
NAME = ""

def run_segmentation(input_image: Image) -> Image:
    return input_image

def benchmark(data_path: str, ground_truth_path: str) -> None:
    data = Dataset()
    data.load_from_path(data_path)
    ground_truth = Dataset()
    ground_truth.load_from_path(ground_truth_path)
    predicted = Dataset()

    if len(data) != len(ground_truth):
        raise Exception(f"Length of data ({len(data)}) and ground_truth ({len(ground_truth)}) do not match!")
    
    timer = TimeBenchmark()

    for i in range(len(data)):
        timer.begin_benchmark()
        
        image: Image = run_segmentation(data[i].image)

        timer.end_benchmark()
        predicted.append(Data(image=image))

    accuracy_benchmark = AccuracyBenchmark(ground_truth, GROUND_TRUTH_LABEL)
    accuracy: float = accuracy_benchmark.get_accuracy(predicted, PREDICTED_LABEL)

    print(f'Time per frame: {timer.get_average_time()}')
    print(f'Accuracy: {accuracy}')
    
