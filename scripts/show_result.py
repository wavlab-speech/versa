import json
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("input", type=str)
args = parser.parse_args()

file = open(args.input, "r", encoding="utf-8")

info = []
for line in file:
    line = line.strip().replace("'", '"').replace("inf", "Infinity")
    info.append(json.loads(line))

keys = info[0].keys()

for key in keys:
    if key == "key":
        continue
    print("key {}: {}".format(key, sum([element[key] for element in info]) / len(info)))
