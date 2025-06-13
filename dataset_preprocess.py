import json 
import argparse


def parse_args():

    parser = argparse.ArgumentParser(prog="Dataset Preprocess", usage='%(prog)s [options]')
    parser.add_argument("--clean_path", type=str, required=True, help="Path to the clean dataset")
    parser.add_argument("--noisy_path", type=str, required=True, help="Path to the noisy dataset")                              
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    