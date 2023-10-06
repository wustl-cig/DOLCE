################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction
################################################################################

import h5py
import numpy as np
import blobfile as bf
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


class dataset2run:

    """
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    def __init__(
        self,
        *,
        data_dir,
        batch_size,
        image_size,
        deterministic=True,
        angle_range=None,
    ):

        if not data_dir:
            raise ValueError("unspecified data directory")

        try:
            all_files = np.load(data_dir).tolist()
        except:        
            all_files = _list_image_files_recursively(data_dir)
        self.all_files = all_files    
        dataset = ImageDataset(
            image_size,
            all_files,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            angle_range=angle_range,
        )
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=1, drop_last=False
        )
    
    def len_data(self):
        return len(self.all_files)

    def load_data(self):
        while True:
            yield from self.loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["h5"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        angle_range=60,
    ):
        super().__init__()
        self.angle_range = angle_range
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        
        with h5py.File(self.local_images[idx], "r") as f:
            data = {}
            for key in f.keys():
                if key != 'Object':
                    data[key] = np.array(f[key]).astype(np.float32)        
                else:
                    img_category = np.array(f[key]).astype(str)

        arr_img512 = data['img']
        
        if img_category == 'COE':
            arr_img512 = arr_img512.clip(0, 1.)
            arr_img512 = arr_img512
        elif img_category =='CKC':
            arr_img512 = arr_img512.clip(0, 0.1)
            arr_img512 = arr_img512/0.35

        angle_range = self.angle_range

        if angle_range == 60:
            arr_la_fbp = data['la60_fbp']
            arr_la_rls = data['la60_rls']
        elif angle_range == 90:
            arr_la_fbp = data['la90_fbp']
            arr_la_rls = data['la90_rls']            
        elif angle_range == 120:
            arr_la_fbp = data['la120_fbp']
            arr_la_rls = data['la120_rls']
        
        arr_la_fbp, arr_la_rls = np.flipud(arr_la_fbp), np.flipud(arr_la_rls)

        arr_la_fbp, arr_la_rls = arr_la_fbp.clip(0, np.inf), arr_la_rls.clip(0, np.inf)

        arr_la_fbp = (arr_la_fbp - arr_la_fbp.min())/(arr_la_fbp.max() - arr_la_fbp.min())
        arr_la_rls = (arr_la_rls - arr_la_rls.min())/(arr_la_rls.max() - arr_la_rls.min())

        arr_img512, arr_la_fbp, arr_la_rls = arr_img512[None,...], arr_la_fbp[None,...], arr_la_rls[None,...]

        out_dict = {}
        out_dict["condition_fbp"], out_dict["condition_rls"] = arr_la_fbp, arr_la_rls
        return arr_img512, out_dict 
