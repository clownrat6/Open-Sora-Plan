import os
import os.path as osp
import math
import glob
import pickle
import random
import warnings

import torch
import decord
import torchvision
import numpy as np
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips
from decord import VideoReader, cpu


decord.bridge.set_bridge('torch')


def build_videoae_dataset(data_folder, sequence_length, resolution, train=True):
    if 'internvid_hr' in data_folder:
        return InternVIDAEDataset(data_folder, sequence_length, resolution)
    elif 'kinetics' in data_folder:
        return Kinetics400Dataset(data_folder, sequence_length, resolution, train)
    else:
        return VideoAEDataset(data_folder, sequence_length, resolution, train)


class Kinetics400Dataset(data.Dataset):
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, resolution=64, train=True):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        if train:
            folder = osp.join(data_folder, 'videos_train')
            file_list = osp.join(data_folder, 'kinetics400_train_list_videos.txt' if train else 'kinetics400_val_list_videos.txt')
        else:
            folder = osp.join(data_folder, 'videos_val')
            file_list = osp.join(data_folder, 'kinetics400_val_list_videos.txt')
        
        self.files = [os.path.join(folder, x.split(' ')[0]) for x in open(file_list, 'r').readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resolution = self.resolution
        # import time
        # s = time.time()
        # decord_vr = VideoReader(self.files[idx], ctx=cpu(0))

        # if len(decord_vr) < self.sequence_length:
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        # start_idx = random.randint(0, len(decord_vr) - self.sequence_length)
        # all_indices = np.arange(start_idx, start_idx + self.sequence_length, 1)
        # video_data = decord_vr.get_batch(all_indices)
        # e = time.time()
        # print(e - s)

        video_path = self.files[idx]

        reader = torchvision.io.VideoReader(video_path, "video", num_threads=16, device="cuda")

        return dict(video=preprocess(video_data, resolution))


class InternVIDAEDataset(data.Dataset):
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.resolution = resolution

        data_folder = osp.join(data_folder, 'nps')

        files = []
        folders = os.listdir(data_folder)
        for folder in folders:
            folder = os.path.join(data_folder, folder)
            print(f"scanning {folder}...")
            _files = glob.glob(os.path.join(folder, f'*.npz'), recursive=True)
            files.extend(_files)
        self.files = files

        self._cache = {}
        self._cache_keys = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resolution = self.resolution
        # load video
        # decord_vr = VideoReader(self.files[idx], ctx=cpu(0))
        # all_indices = np.arange(0, len(decord_vr), 1)
        # video_data = decord_vr.get_batch(all_indices).asnumpy()
        # video_data = torch.from_numpy(video_data)
        key = random.choice(['arr_0', 'arr_1', 'arr_2'])

        try:
            file_obj = np.load(self.files[idx])
        except Exception as e:
            print(f"Error in loading {self.files[idx]}: {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        video_data = file_obj[key]
        video_data = torch.from_numpy(video_data)

        # cache_key = os.path.basename(self.files[idx]) + '$' + key
        # if cache_key in self._cache_keys:
        #     video_data = self._cache[cache_key]
        # else:
        #     try:
        #         file_obj = np.load(self.files[idx])
        #     except Exception as e:
        #         print(f"Error in loading {self.files[idx]}: {e}")
        #         return self.__getitem__(random.randint(0, self.__len__() - 1))
        #     video_data = file_obj[key]
        #     video_data = torch.from_numpy(video_data)
        #     self._cache[cache_key] = video_data
        #     self._cache_keys.append(cache_key)
        #     if len(self._cache_keys) > 100000:
        #         del self._cache[self._cache_keys.pop(0)]

        # file_obj = np.load(self.files[idx])
        # keys = list(file_obj.keys())
        # video_data = file_obj[random.choice(keys)]
        # video_data = torch.from_numpy(video_data)

        try:
            ret = dict(video=preprocess(video_data, resolution, self.sequence_length))
        except AssertionError as e:
            print(f"Error in processing {self.files[idx]} which does not have enough frames.")
            ret = self.__getitem__(random.randint(0, self.__len__() - 1))
        except RuntimeError as e:
            print(f"Error in processing {self.files[idx]}.")
            ret = self.__getitem__(random.randint(0, self.__len__() - 1))

        return ret


# Copied from https://github.com/wilson1yan/VideoGPT
class VideoAEDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, resolution=64, train=True):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            if dist.is_initialized() and dist.get_rank() == 0:
                pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(video=preprocess(video, resolution), label=label)


# Copied from https://github.com/wilson1yan/VideoGPT
def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


# Copied from https://github.com/wilson1yan/VideoGPT
def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)
    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW
    video -= 0.5

    return video
