from io import BytesIO
# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util


class Defocus_dataset(Dataset):
    def __init__(self, dataroot, datatype, origin_fnumber=16, target_fnumber=2, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.ori_fn = origin_fnumber
        self.tar_fn = target_fnumber
        self.data_len = data_len
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/FNumber_{}'.format(dataroot, origin_fnumber))
            self.hr_path = Util.get_paths_from_images(
                '{}/FNumber_{}'.format(dataroot, target_fnumber))

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_tar = None
        img_ori = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                tar_img_bytes = txn.get(
                    'FNumber_{}_{}'.format(
                        self.tar_fn, str(index).zfill(5)).encode('utf-8')
                )
                ori_img_bytes = txn.get(
                    'sr_{}_{}'.format(
                        self.ori_fn, str(index).zfill(5)).encode('utf-8')
                )
                # skip the invalid index
                while (tar_img_bytes is None) or (ori_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    tar_img_bytes = txn.get(
                        'FNumber_{}_{}'.format(
                            self.tar_fn, str(new_index).zfill(5)).encode('utf-8')
                    )
                    ori_img_bytes = txn.get(
                        'FNumber_{}_{}'.format(
                            self.ori_fn, str(new_index).zfill(5)).encode('utf-8')
                    )
                img_tar = Image.open(BytesIO(tar_img_bytes)).convert("RGB")
                img_ori = Image.open(BytesIO(ori_img_bytes)).convert("RGB")

        else:
            img_tar = Image.open(self.hr_path[index]).convert("RGB")
            img_ori = Image.open(self.sr_path[index]).convert("RGB")

        [img_ori, img_tar] = Util.transform_augment(
            [img_ori, img_tar], split=self.split, min_max=(-1, 1))
        return {'Gt': img_tar, 'Cond': img_ori, 'Index': index}
