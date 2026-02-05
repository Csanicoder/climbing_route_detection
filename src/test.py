import os.path
from pathlib import Path

path : Path = "../data/"

file = "black_v4"
ext = "_hold.json"

print(os.path.join(path, file + ext))