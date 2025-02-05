import os
import json
import h5py
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

@register_dataset("anet")
class ActivityNetDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        definition_file,  # json file for definition
        zeroshot_split,   # zero-shot setting
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        use_definition,
        data_percent
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.use_hdf5 = '.hdf5' in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        self.zeroshot_split = zeroshot_split
        if self.zeroshot_split != "None":
            percentage = int(self.zeroshot_split[0])
            mode = "train" if self.is_training else "test"
            split_file_path = os.path.join("splits", "train{:d}_test{:d}".format(percentage, 100 - percentage),
                                           "ActivityNet1.3", mode, "{}.list".format(self.zeroshot_split[1]))
            self.zeroshot_classes = list()
            with open(split_file_path, "r") as fp:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    label = line.rstrip()
                    self.zeroshot_classes.append(label)
        else:
            self.zeroshot_classes = None

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        if self.zeroshot_split != "None":
            self.zeroshot_label_indices = sorted([label_dict[label] for label in self.zeroshot_classes])
        else:
            self.zeroshot_label_indices = None
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        if is_training and data_percent < 1.0:
            self.data_list = random.sample(self.data_list, round(len(self.data_list) * float(data_percent)))
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }

        self.use_definition = use_definition
        with open(definition_file, 'r') as fid:
            self.definition = json.load(fid)

        self.prompt_templates = ["an action of"]
        # self.prompt_templates = ["an action of", "actions of", "a human action of", "human actions of",
        #                          "an activity of", "a human activity of", "a clip of", "clips of",
        #                          "a clip showing", "clips showing", "a video of", "videos of",
        #                          "a video showing", "videos showing"]

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                valid_num = 0

                if num_acts:
                    target_query = random.choice([act['label'] for act in valid_acts])
                    for idx, act in enumerate(valid_acts):
                        query = act['label']
                        # if query != target_query:
                        #     continue
                        segments[idx][0] = act['segment'][0]
                        segments[idx][1] = act['segment'][1]
                        labels[idx] = label_dict[query]
                        valid_num += 1

                    if valid_num:
                        dict_db += ({'id': key,
                                     'fps': fps,
                                     'duration': duration,
                                     'segments': segments,
                                     'labels': labels,
                                     'queries': target_query.lower()
                                     },)

            # if ('annotations' in value) and (len(value['annotations']) > 0):
            #     valid_acts = remove_duplicate_annotations(value['annotations'])
            #     num_acts = len(valid_acts)
            #     segments = np.zeros([num_acts, 2], dtype=np.float32)
            #     labels = np.zeros([num_acts, ], dtype=np.int64)
            #     for idx, act in enumerate(valid_acts):
            #         segments[idx][0] = act['segment'][0]
            #         segments[idx][1] = act['segment'][1]
            #         if self.num_classes == 1:
            #             labels[idx] = 0
            #         else:
            #             labels[idx] = label_dict[act['label']]
            #     queries = act['label'].lower()
            # else:
            #     segments = None
            #     labels = None
            # dict_db += ({'id': key,
            #              'fps': fps,
            #              'duration': duration,
            #              'segments': segments,
            #              'labels': labels,
            #              'queries': queries
            #              },)

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        if self.use_hdf5:
            with h5py.File(self.feat_folder, 'r') as h5_fid:
                feats = np.asarray(
                    h5_fid[self.file_prefix + video_item['id']][()],
                    dtype=np.float32
                )
        else:
            filename = os.path.join(self.feat_folder,
                                    self.file_prefix + video_item['id'] + self.file_ext)
            feats = np.load(filename).astype(np.float32)

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            # segments = torch.from_numpy(video_item['segments'] / video_item['duration'])
            if self.num_classes == 1:
                labels = torch.zeros_like(torch.from_numpy(video_item['labels']))
            queries = video_item['queries']
            tgt_label = queries

            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                            (min(seg[1].item(), vid_len) - seg[0].item())
                            / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels, queries = None, None, None

        # queries = feats.mean(-1)
        queries = None
        # queries = ["all actions"]

        # basic = "all actions"
        # condition = torch.zeros(dtype=torch.float32, size=(1 + 4 + 1 + 12 * 2, ))

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'queries'         : queries,    # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict