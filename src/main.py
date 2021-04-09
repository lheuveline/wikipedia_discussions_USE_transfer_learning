import argparse

import extract
import train

parser = argparse.ArgumentParser()
parser.add_argument("--extract", action='store_true')
parser.add_argument("--train", action='store_true')
parser = parser.parse_args()

def main():

    if parser.extract:
        extract.main()

    if parser.train:
        train.main()


if __name__ == "__main__":
    main()
