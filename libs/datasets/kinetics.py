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

@register_dataset("kinetics")
class KineticsDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        definition_file,  # json file for definition
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
        force_upsampling, # force to upsample to max_seq_len
        use_definition
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

        # load database and select the subset
        # dict_db = self._load_json_db(self.json_file)
        # proposal vs action categories
        # assert (num_classes == 1) or (len(label_dict) == num_classes)
        with open(json_file, 'r') as fid:
            self.database = json.load(fid)
        self.data_list = list(self.database.keys())
        self.data_list = [key for key in self.database.keys()
                          if "all" in self.split or self.database[key]["subset"] in self.split]
        # self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Kinetics-400',
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

        self.ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                        "eighth", "ninth", "tenth", "eleventh", "twelfth"]

        label_keys = sorted(self.definition.keys())
        self.label_dict = dict()
        self.label_inverse_dict = dict()
        for i, key in enumerate(label_keys):
            self.label_dict[key] = i
            self.label_inverse_dict[i] = key

        self.class_sim = np.load("./definition/semantic_similarity.npy")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            print(self.split)
            # skip the video if not in the split
            if "all" not in self.split and value['subset'].lower() not in self.split:
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
            label = value['label']
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'label' : label
            }, )

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        while True:
            tgt_video_key = random.choice(self.data_list)
            if os.path.exists(os.path.join(self.feat_folder, self.file_prefix + tgt_video_key + self.file_ext)):
                tgt_video_item = self.database[tgt_video_key]
                tgt_video_item["id"] = tgt_video_key
                break

        tgt_label = tgt_video_item["label"]

        other_class_len = self.max_seq_len // 16
        same_class_len = random.choice(range(1, self.max_seq_len // 16 + 1))
        same_class_videos = list()
        while True:
            sampled_key = random.choice(self.data_list)
            sampled_item = self.database[sampled_key]
            if sampled_item["label"] == tgt_label and \
                    os.path.exists(os.path.join(self.feat_folder, self.file_prefix + sampled_key + self.file_ext)):
                sampled_item["id"] = sampled_key
                same_class_videos.append(sampled_item)
            if len(same_class_videos) >= same_class_len:
                break

        # top_k_other_classes = np.argsort(-self.class_sim[self.label_dict[tgt_label]])[1:1 + 10]

        other_class_videos = list()
        while True:
            sampled_key = random.choice(self.data_list)
            sampled_item = self.database[sampled_key]
            # if sampled_item["label"] != tgt_label and \
            #         self.label_dict[sampled_item["label"]] in top_k_other_classes and \
            #         os.path.exists(os.path.join(self.feat_folder, self.file_prefix + sampled_key + self.file_ext)):
            if sampled_item["label"] != tgt_label and \
                    os.path.exists(os.path.join(self.feat_folder, self.file_prefix + sampled_key + self.file_ext)):
                sampled_item["id"] = sampled_key
                other_class_videos.append(sampled_item)
            if len(other_class_videos) >= other_class_len:
                break

        feats = list()
        targets = list()
        for item in other_class_videos:
            this_feats = np.load(os.path.join(self.feat_folder, self.file_prefix + item['id'] + self.file_ext),
                                 allow_pickle=True)
            len_feat = len(this_feats)
            feats.append(this_feats)
            targets.append(np.asarray([self.label_dict[item["label"]]] * len_feat))
        feats = np.concatenate(feats, axis=0)
        targets = np.concatenate(targets, axis=0)

        # feats = np.load(os.path.join(self.feat_folder, self.file_prefix + tgt_video_item["id"] + self.file_ext))
        # tgt_size = random.choice(range(1, len(feats) + 1))
        # src_s_i = random.choice(range(0, len(feats) - tgt_size + 1))
        # targets = np.zeros(dtype=np.int32, shape=(len(feats), ))
        # targets[:] = self.num_classes
        # targets[src_s_i:src_s_i + tgt_size] = self.label_dict[tgt_label]
        # queries = torch.from_numpy(np.mean(feats[src_s_i:src_s_i + tgt_size], axis=0))

        if len(feats) != self.max_seq_len:
            feats = F.interpolate(
                torch.from_numpy(feats.transpose(1, 0)[None]),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False)
            feats = feats.squeeze(0).transpose(1, 0).numpy()

            targets = F.interpolate(
                torch.from_numpy(targets[:, None].transpose(1, 0)[None]).float(),
                size=self.max_seq_len,
                mode='nearest')
            targets = targets.int().squeeze(0).transpose(1, 0).numpy().squeeze(-1)

        boundaries = np.linspace(0, self.max_seq_len, same_class_len + 1, dtype=np.int32)
        for b_i in range(len(boundaries) - 1):
            item = same_class_videos[b_i]
            src_feats = np.load(os.path.join(self.feat_folder, self.file_prefix + item['id'] + self.file_ext), allow_pickle=True)
            tgt_size = random.choice(range(1, len(src_feats) + 1))
            src_s_i = random.choice(range(0, len(src_feats) - tgt_size + 1))
            tgt_s_i = random.choice(range(boundaries[b_i], boundaries[b_i + 1] - tgt_size + 1))
            for f_i in range(tgt_size):
                feats[(tgt_s_i + f_i) % len(feats)] = src_feats[(src_s_i + f_i) % len(src_feats)]
                targets[(tgt_s_i + f_i) % len(targets)] = self.label_dict[item["label"]]

        all_slices = list()
        this_target_sequence = targets.tolist()
        run = {"class_number": this_target_sequence[0], "indices": []}
        slices, expect = [run], None
        for index, target in enumerate(this_target_sequence):
            if (target == expect) or (expect is None):
                run["indices"].append(index)
            else:
                run = {"class_number": target, "indices": [index]}
                slices.append(run)
            expect = target

        for slice in slices:
            if slice["class_number"] >= 0:
                all_slices.append([slice["indices"][0], slice["indices"][-1], slice["class_number"]])

        # semantic_condition = random.random() > 0.5
        # if semantic_condition and len([slice for slice in target_slices if slice[-1] != self.label_dict[tgt_label]]):
        #     # target_slices = [slice for slice in target_slices if slice[-1] != self.label_dict[tgt_label]]
        #     # tgt_label = "all actions except for {}".format(tgt_label)
        #     tgt_label = "all actions"
        # else:
        #     target_slices = [slice for slice in target_slices if slice[-1] == self.label_dict[tgt_label]]
        #     tgt_label = "all actions of {}".format(tgt_label)

        # other_slices = [slice for slice in all_slices if slice[-1] != self.label_dict[tgt_label]]
        target_slices = [slice for slice in all_slices if slice[-1] == self.label_dict[tgt_label]]

        if self.use_definition:
            definition = self.definition[tgt_label]
            tgt_label = tgt_label + ", " + definition

        # temporal_condition = random.random() > 0.5
        # if temporal_condition:
        #     sampled_start = random.choice(range(len(target_slices)))
        #     sampled_len = random.choice(range(len(target_slices) - sampled_start)) + 1
        #     target_slices = target_slices[sampled_start:sampled_start + sampled_len]
        #     tgt_label = "{} to {} actions of {}".format(self.ordinal[sampled_start],
        #                                                 self.ordinal[sampled_start + sampled_len - 1],
        #                                                 tgt_label)
        # else:
        #     tgt_label = "all actions of {}".format(tgt_label)

        # temporal_condition = random.random() > 0.5
        # if temporal_condition:
        #     sampled_slice = random.choice(target_slices)
        #     sampled_length = (sampled_slice[1] - sampled_slice[0] + 1) / self.max_seq_len
        #     if sampled_length < 0.33:
        #         tgt_label = "short actions of {}".format(tgt_label)
        #         target_slices = [slice for slice in target_slices if (slice[1] - slice[0]) / self.max_seq_len < 0.33]
        #     elif sampled_length < 0.66:
        #         tgt_label = "medium actions of {}".format(tgt_label)
        #         target_slices = [slice for slice in target_slices if 0.33 < (slice[1] - slice[0]) / self.max_seq_len < 0.66]
        #     else:
        #         tgt_label = "long actions of {}".format(tgt_label)
        #         target_slices = [slice for slice in target_slices if (slice[1] - slice[0]) / self.max_seq_len > 0.66]
        # else:
        #     tgt_label = "all actions of {}".format(tgt_label)

        split = 0
        # is_conditioned = False
        is_conditioned = random.random() > 0.5
        # is_conditioned = random.random() >= 0.0

        condition_type = random.choice(["scale"])
        # condition_type = random.choice(["ordinal"])
        # condition_type = random.choice(["all"])
        # condition_type = random.choice(["scale", "ordinal"])
        # condition_type = random.choice(["scale", "ordinal", "all"])
        split = 0

        class_label = torch.zeros(dtype=torch.float32, size=(400,))
        class_label[self.label_dict[tgt_label]] = 1.0
        scale_label = torch.zeros(dtype=torch.float32, size=(1 + 4, ))
        scale_label[0] = 1.0
        ordinal_label = torch.zeros(dtype=torch.float32, size=(1 + other_class_len * 2, ))
        ordinal_label[0] = 1.0
        if is_conditioned:
            if condition_type in ["ordinal", "all"]:
                sampled_start = random.choice(range(len(target_slices)))
                sampled_len = random.choice(range(len(target_slices) - sampled_start)) + 1
                target_slices = target_slices[sampled_start:sampled_start + sampled_len]
                # ordinal_condition_label = "{} to {}".format(self.ordinal[sampled_start],
                #                                             self.ordinal[sampled_start + sampled_len - 1])
                # if condition_type == "all":
                #     condition_label = condition_label + " " + ordinal_condition_label
                # else:
                #     condition_label = ordinal_condition_label

                # tgt_label = "{} to {} actions of {}".format(self.ordinal[sampled_start],
                #                                             self.ordinal[sampled_start + sampled_len - 1],
                #                                             tgt_label)
                ordinal_label[0] = 0.0
                ordinal_label[1 + sampled_start] = 1.0
                ordinal_label[1 + other_class_len + sampled_start + sampled_len - 1] = 1.0

                # while True:
                #     other_slice = random.choice(other_slices)
                #     other_label = self.label_inverse_dict[other_slice[-1]]
                #     direction = random.choice(["before", "after"])
                #     target_slices = [slice for s_i, slice in enumerate(all_slices) if slice[-1] == self.label_dict[tgt_label]
                #                      and (((direction == "before" and s_i < len(all_slices) - 1) and all_slices[s_i + 1][-1] == other_slice[-1]) or
                #                           (((direction == "after" and s_i >= 1) and all_slices[s_i - 1][-1] == other_slice[-1])))]
                #     if len(target_slices):
                #         tgt_label = "all actions of {}, {} {}".format(tgt_label, direction, other_label)
                #         break

            if condition_type in ["scale", "all"]:
                scale_label[0] = 0.0
                sampled_slice = random.choice(target_slices)
                sampled_length = (sampled_slice[1] - sampled_slice[0] + 1) / self.max_seq_len
                if sampled_length < 0.25:
                    scale_label[1 + 0] = 1.0
                    condition_label = "extra-short-duration"
                    # tgt_label = "etra-short-duration actions of {}".format(tgt_label)
                    target_slices = [slice for slice in target_slices if (slice[1] - slice[0] + 1) / self.max_seq_len < 0.25]
                elif sampled_length < 0.50:
                    scale_label[1 + 1] = 1.0
                    condition_label = "short-duration"
                    # tgt_label = "short-duration actions of {}".format(tgt_label)
                    target_slices = [slice for slice in target_slices if 0.25 <= (slice[1] - slice[0] + 1) / self.max_seq_len < 0.50]
                elif sampled_length < 0.75:
                    scale_label[1 + 2] = 1.0
                    condition_label = "long-duration"
                    # tgt_label = "long-duration actions of {}".format(tgt_label)
                    target_slices = [slice for slice in target_slices if 0.50 <= (slice[1] - slice[0] + 1) / self.max_seq_len < 0.75]
                else:
                    scale_label[1 + 3] = 1.0
                    condition_label = "extra-long-duration"
                    # tgt_label = "extra-long-duration actions of {}".format(tgt_label)
                    target_slices = [slice for slice in target_slices if (slice[1] - slice[0] + 1) / self.max_seq_len >= 0.75]
        else:
            condition_label = "all"
            # tgt_label = "all actions of {}".format(tgt_label)

        target_slices = np.asarray(target_slices, dtype=np.float32)

        segments = target_slices[:, :2] / float(self.max_seq_len - 1)
        # segments = torch.from_numpy(
        #     video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
        # )
        # deal with fixed length feature, recompute feat_stride, num_frames

        fps = self.default_fps
        duration = other_class_len * 10.0
        seq_len = self.max_seq_len
        feat_stride = duration * fps / seq_len
        # center the features
        num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        segments = segments * duration
        segments = segments * fps / feat_stride - feat_offset
        labels = np.zeros_like(target_slices[:, -1].astype(np.int64))

        # tgt_label = "{} actions of {}".format(condition_label, tgt_label)
        # queries = [tgt_label]
        # queries = torch.cat((class_label, scale_label, ordinal_label), dim=0)
        queries = "all actions of {}".format(tgt_label)
        conditions = torch.cat((scale_label, ordinal_label), dim=0)

        # queries = tgt_label
        # queries = np.asarray(["{} {}".format(prompt, tgt_label) for prompt in self.prompt_templates])
        # queries = np.load(os.path.join(self.feat_folder, self.file_prefix + random.choice(same_class_videos)['id'] +
        #                                self.file_ext)).mean(axis=0)
        # queries = torch.from_numpy(queries)
        # queries = F.one_hot(torch.Tensor([self.label_dict[tgt_label]]).long(), num_classes=400).float().squeeze(0)

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        segments = torch.from_numpy(segments)
        labels = torch.from_numpy(labels)

        # return a data dict
        data_dict = {'video_id'        : tgt_video_key,
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'queries'         : queries,    # N
                     'conditions'      : conditions,    # N
                     'fps'             : self.default_fps,
                     'duration'        : duration,
                     'feat_stride': feat_stride,
                     'feat_num_frames': num_frames
                     }

        # no truncation is needed
        # truncate the features during training
        # if self.is_training and (segments is not None):
        #     data_dict = truncate_feats(
        #         data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
        #     )

        return data_dict