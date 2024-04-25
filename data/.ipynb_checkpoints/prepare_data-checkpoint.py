import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def data_get(img_path, img_file, FNumber):
    img = Image.open(img_file)
    img = img.convert('RGB')

    return img_file.name.split('.')[0]

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
        hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))


def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, FNumber=(16, 2), lmdb_save=False):
    resize_fn = partial(data_get, FNumber = FNumber, lmdb_save=lmdb_save)
    files = os.listdir(img_path)


    os.makedirs(out_path, exist_ok=True)
    os.makedirs('{}/FNumber_{}'.format(out_path, FNumber[0]), exist_ok=True)
    os.makedirs('{}/FNumber_{}'.format(out_path, FNumber[1]), exist_ok=True)


    if n_worker > 1:
        # prepare data subsets
        multi_env = None

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, FNumber)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img = imgs

            lr_img.save(
                    '{}/FNumber_{}/{}.png'.format(out_path, FNumber[0], i.zfill(5)))
            hr_img.save(
                    '{}/Fnumber_{}/{}.png'.format(out_path, FNumber[1], i.zfill(5)))

            total += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '-p', type=str,
    #                     default='{}/Dataset/defocus'.format(Path.home()))
    parser.add_argument('--path', '-p', type=str,
                         default='/mnt/data0/ICME_dataset/defocus_train_test_choose/train')
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/defocus')

    parser.add_argument('--FNumber', type=str, default='16,2')
    parser.add_argument('--n_worker', type=int, default=0)
    # parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    # resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    # resample = resample_map[args.resample]
    FNumber = [int(s.strip()) for s in args.size.split(',')]

    # args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    # prepare(args.path, args.out, args.n_worker,
    #         sizes=sizes, resample=resample, lmdb_save=args.lmdb)
    prepare(args.path, args.out, args.n_worker, FNumber, lmdb_save=args.lmdb)

