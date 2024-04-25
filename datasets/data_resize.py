import os
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import argparse

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_number(file_name):
    return int(file_name.stem.split(".")[0])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='./validation2')
    parser.add_argument('-o', '--outpath', type=str,
                        default= './resizedvalidation')
    args = parser.parse_args()

    path = Path(args.path)
    outpath = Path(args.outpath)

    F2_path = path / 'FNumber_2'
    F4_path = path / 'FNumber_16'


    F2_outpath = outpath / 'FNumber_2'
    F4_outpath = outpath / 'FNumber_16'


    F2_outpath.mkdir(parents = True, exist_ok = True)
    F4_outpath.mkdir(parents = True, exist_ok = True)

    F2_list = list(F2_path.glob("*.jpg"))
    F4_list = list(F4_path.glob("*.jpg"))


    F2_list.sort(key= get_number)
    F4_list.sort(key= get_number)


    count = 0
    for i in range(0, len(F2_list)):
        assert(F2_list[i].name == F4_list[i].name), ('please check dataset')

        F2_img = Image.open(F2_list[i])
        F4_img = Image.open(F4_list[i])

        F2_resize = F2_img.resize((1152, 768), resample=Image.LANCZOS) 
        F4_resize = F4_img.resize((1152, 768), resample=Image.LANCZOS) 
        F2_resize.save('{}/{}.png'.format(str(F2_outpath), str(count).zfill(4)))
        F4_resize.save('{}/{}.png'.format(str(F4_outpath), str(count).zfill(4)))

        count = count + 1