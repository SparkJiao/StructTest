import csv
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()

if os.path.exists(args.input_file):
    files = [args.input_file]
else:
    files = list(glob(args.input_file))

for file in files:
    if not file.endswith(".csv"):
        continue
    tot = 0
    cnt = 0
    with open(file, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            tot += 1
            if eval(row["result"]):
                cnt += 1
    print(f"{file}: {cnt}/{tot} ({cnt / tot:.4f})")
