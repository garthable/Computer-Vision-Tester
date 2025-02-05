import argparse
from benchmark import benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-g", "--ground_truth", required=True,
        help="path to input ground truth data")
    args = vars(ap.parse_args())

    benchmark(args["dataset"], args["ground_truth"])