import sys
from pathlib import Path

import numpy as np
from imageio import imread, imwrite

from depth_utils import gray2rgb

if __name__ == '__main__':
    dirs = [Path(sys.argv[1])]
    while len(dirs) > 0:
        dir = dirs[0]
        dirs = dirs[1:]
        for f in dir.iterdir():
            if f.is_file():
                if f.name.split(".")[-1] == "png" and "plasma" not in f.name:
                    new_name = f.name.replace(".png", "_plasma.png")
                    data = imread(str(f)).astype('float32')
                    data /= data.max()
                    imwrite(str(f.parent / new_name), gray2rgb(data, "plasma"))

            elif f.is_dir():
                dirs.append(f)
