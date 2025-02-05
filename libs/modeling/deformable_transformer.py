# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from ..utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=True,
                 num_classes=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.box_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        encoder_layer = DeformableTransformerCrossEncoderLayer(d_model, dim_feedforward,
                                                               dropout, activation,
                                                               num_feature_levels, nhead, enc_n_points)
        self.box_cross_encoder = DeformableTransformerCrossEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    use_dab=use_dab, d_model=d_model,
                                                    high_dim_query_update=high_dim_query_update,
                                                    no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # if two_stage:
        #     self.enc_output = nn.Linear(d_model, d_model)
        #     self.enc_output_norm = nn.LayerNorm(d_model)
        #     self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        #     self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        # else:
        #     if not self.use_dab:
        #         self.reference_points = nn.Linear(d_model, 2)

        self.num_classes = num_classes

        if two_stage:
            self.proposal_embed = nn.Sequential(
                nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, 3, kernel_size=1))

            self.video_embed = nn.Sequential(
                nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, self.num_classes, kernel_size=1))

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_1d_embeds, pos_2d_embeds,
                box_srcs, box_pos_1d_embeds, box_pos_2d_embeds,
                query_embed=None, attn_mask=None, label_enc=None):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        lvl_pos_1d_embed_flatten = []
        lvl_pos_2d_embed_flatten = []
        spatial_shapes_1d = []
        spatial_shapes_2d = []
        for lvl, (src, pos_1d_embed, pos_2d_embed) in enumerate(zip(srcs, pos_1d_embeds, pos_2d_embeds)):
            bs, c, h, w = src.shape
            spatial_shape_1d = (h, w)
            spatial_shape_2d = (h, h)
            spatial_shapes_1d.append(spatial_shape_1d)
            spatial_shapes_2d.append(spatial_shape_2d)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_1d_embed = pos_1d_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_2d_embed = pos_2d_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_1d_embed = pos_1d_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_2d_embed = pos_2d_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_1d_embed_flatten.append(lvl_pos_1d_embed)
            lvl_pos_2d_embed_flatten.append(lvl_pos_2d_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        lvl_pos_1d_embed_flatten = torch.cat(lvl_pos_1d_embed_flatten, 1)
        lvl_pos_2d_embed_flatten = torch.cat(lvl_pos_2d_embed_flatten, 1)
        spatial_shapes_1d = torch.as_tensor(spatial_shapes_1d, dtype=torch.long, device=src_flatten.device)
        spatial_shapes_2d = torch.as_tensor(spatial_shapes_2d, dtype=torch.long, device=src_flatten.device)
        level_start_index_1d = torch.cat((spatial_shapes_1d.new_zeros((1,)), spatial_shapes_1d.prod(1).cumsum(0)[:-1]))
        level_start_index_2d = torch.cat((spatial_shapes_2d.new_zeros((1,)), spatial_shapes_2d.prod(1).cumsum(0)[:-1]))

        box_src_flatten = []
        box_lvl_pos_1d_embed_flatten = []
        box_lvl_pos_2d_embed_flatten = []
        box_spatial_shapes_1d = []
        box_spatial_shapes_2d = []
        for lvl, (src, pos_1d_embed, pos_2d_embed) in enumerate(zip(box_srcs, box_pos_1d_embeds, box_pos_2d_embeds)):
            bs, c, h, w = src.shape
            spatial_shape_1d = (h, w)
            spatial_shape_2d = (h, h)
            box_spatial_shapes_1d.append(spatial_shape_1d)
            box_spatial_shapes_2d.append(spatial_shape_2d)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_1d_embed = pos_1d_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_2d_embed = pos_2d_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_1d_embed = pos_1d_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_2d_embed = pos_2d_embed + self.level_embed[lvl].view(1, 1, -1)
            box_lvl_pos_1d_embed_flatten.append(lvl_pos_1d_embed)
            box_lvl_pos_2d_embed_flatten.append(lvl_pos_2d_embed)
            box_src_flatten.append(src)
        box_src_flatten = torch.cat(box_src_flatten, 1)  # bs, \sum{hxw}, c
        box_lvl_pos_1d_embed_flatten = torch.cat(box_lvl_pos_1d_embed_flatten, 1)
        box_lvl_pos_2d_embed_flatten = torch.cat(box_lvl_pos_2d_embed_flatten, 1)
        box_spatial_shapes_1d = torch.as_tensor(box_spatial_shapes_1d, dtype=torch.long, device=box_src_flatten.device)
        box_spatial_shapes_2d = torch.as_tensor(box_spatial_shapes_2d, dtype=torch.long, device=box_src_flatten.device)
        box_level_start_index_1d = torch.cat((box_spatial_shapes_1d.new_zeros((1,)),
                                              box_spatial_shapes_1d.prod(1).cumsum(0)[:-1]))
        box_level_start_index_2d = torch.cat((box_spatial_shapes_2d.new_zeros((1,)),
                                              box_spatial_shapes_2d.prod(1).cumsum(0)[:-1]))

        # encoder
        memory = src_flatten
        # memory = self.encoder(src_flatten, spatial_shapes_1d, level_start_index_1d, lvl_pos_1d_embed_flatten)
        box_memory = self.box_encoder(box_src_flatten, box_spatial_shapes_1d,
                                      box_level_start_index_1d, box_lvl_pos_1d_embed_flatten)
        # memory = self.encoder(src_flatten, spatial_shapes_1d, level_start_index_1d, lvl_pos_1d_embed_flatten)
        # box_memory = self.box_cross_encoder(box_src_flatten, memory, box_spatial_shapes_1d,
        #                                     box_level_start_index_1d, box_lvl_pos_1d_embed_flatten)

        memory_2d = list()
        box_memory_2d = list()
        for l_i in range(len(srcs)):
            h, w = spatial_shapes_1d[l_i]
            level_start_index = level_start_index_1d[l_i]
            level_end_index = level_start_index + h * w
            this_memory = memory[:, level_start_index:level_end_index]
            this_memory_2d = this_memory.unsqueeze(2).repeat(1, 1, h, 1).flatten(1, 2)
            memory_2d.append(this_memory_2d)
            this_box_memory = box_memory[:, level_start_index:level_end_index]
            this_box_memory_2d = this_box_memory.unsqueeze(2).repeat(1, 1, h, 1).flatten(1, 2)
            box_memory_2d.append(this_box_memory_2d)
        memory_2d = torch.cat(memory_2d, 1)
        box_memory_2d = torch.cat(box_memory_2d, 1)

        # if self.two_stage:
        #     target_length = encoder_outputs[-1].shape[1]
        #     merged_encoder_outputs = list()
        #     for e_o in encoder_outputs:
        #         _, t, _ = e_o.shape
        #         if t != target_length:
        #             e_o = F.interpolate(e_o.permute(0, 2, 1), size=target_length, mode="linear")
        #         else:
        #             e_o = e_o.permute(0, 2, 1)
        #         merged_encoder_outputs.append(e_o)
        #     merged_encoder_outputs = torch.cat(merged_encoder_outputs, dim=1)
        #
        #     proposals = self.proposal_embed(merged_encoder_outputs).permute(0, 2, 1)
        #     bs, t, _ = proposals.shape
        #     _, q, _ = query_embed.shape
        #     q = q // 4
        #     proposal_boxes = proposals.detach()[..., 1:].sigmoid()
        #     proposal_boxes[..., 0] = -proposal_boxes[..., 0]
        #     proposal_boxes = torch.arange(t, dtype=torch.float32, device=proposals.device)[None, :, None] / (t - 1) + \
        #                      proposal_boxes
        #     proposal_boxes = torch.clamp(proposal_boxes, 0.0, 1.0)
        #     proposal_boxes = torch.stack((proposal_boxes[..., 0], proposal_boxes[..., 1],
        #                                   (proposal_boxes[..., 0] + proposal_boxes[..., 1]) / 2.0,
        #                                   proposal_boxes[..., 1] - proposal_boxes[..., 0]), dim=-1)
        #     topk_indices = torch.topk(proposals.detach()[..., 0], k=q, dim=1)[1]
        #     topk_references = torch.gather(proposal_boxes, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        #
        #     video_embeds = self.video_embed(merged_encoder_outputs).permute(0, 2, 1).mean(dim=1)
        #     video_class = torch.argmax(video_embeds.detach(), dim=-1)
        #     video_embeddings = label_enc(video_class).unsqueeze(1).repeat(1, q, 1)
        # else:
        #     proposals = None
        #     video_embeds = None

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid()
            tgt = query_embed[..., :self.d_model]
            # if self.two_stage:
            #     reference_points = torch.cat((reference_points[:, :-q], topk_references), dim=1)
            #     tgt_new = torch.cat([video_embeddings, torch.zeros([bs, q, 1], device=video_embeddings.device)], dim=-1)
            #     tgt = torch.cat((tgt[:, :-q], tgt_new), dim=1)
            # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            # bs, num_quires, 2
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory_2d,
                                            lvl_pos_2d_embed_flatten, spatial_shapes_2d, level_start_index_2d,
                                            box_memory_2d, box_lvl_pos_2d_embed_flatten,
                                            box_spatial_shapes_2d, box_level_start_index_2d,
                                            query_pos=query_embed if not self.use_dab else None, attn_mask=attn_mask)

        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (H_)
            ref_x = ref_x.reshape(-1)[None] / (W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerCrossEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, memory, src_pos, memory_pos, reference_points,
                spatial_shapes, level_start_index, padding_mask=None):
        # # self attention
        # src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
        #                       padding_mask)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)

        # cross attention
        src2 = self.cross_attn(self.with_pos_embed(src, src_pos), reference_points,
                               self.with_pos_embed(memory, memory_pos),
                               spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerCrossEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (H_)
            ref_x = ref_x.reshape(-1)[None] / (W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, src, src2, src_spatial_shapes, src2_spatial_shapes, level_start_index, pos=None, memory_pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()

        reference_points = self.get_reference_points(src_spatial_shapes, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, src2, pos, memory_pos,
                           reference_points, src2_spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn_2 = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1_2 = nn.Dropout(dropout)
        self.norm1_2 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_pos, src_spatial_shapes, level_start_index,
                box_src, box_src_pos, box_src_spatial_shapes, box_level_start_index,
                src_padding_mask=None, box_src_padding_mask=None, self_attn_mask=None):\

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self_attn_mask)[
            0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # # cross attention
        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
        #                          reference_points,
        #                          self.with_pos_embed(src, box_features + src_pos),
        #                          src_spatial_shapes, level_start_index, src_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        tgt2 = self.cross_attn_2(self.with_pos_embed(tgt, query_pos),
                                 reference_points,
                                 self.with_pos_embed(box_src, box_src_pos),
                                 box_src_spatial_shapes, box_level_start_index, box_src_padding_mask)
        tgt = tgt + self.dropout1_2(tgt2)
        tgt = self.norm1_2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256,
                 high_dim_query_update=False, no_sine_embed=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)

    def forward(self, tgt, reference_points, src, src_pos, src_spatial_shapes, src_level_start_index,
                box_src, box_src_pos, box_src_spatial_shapes, box_src_level_start_index,
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        # bs = src.shape[0]
        # reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None]  # bs, nq, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None]
            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input[:, :, 0, :])
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # bs, nq, 256*2
                    raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)

            output = layer(output, query_pos, reference_points_input, src, src_pos,
                           src_spatial_shapes, src_level_start_index,
                           box_src, box_src_pos, box_src_spatial_shapes, box_src_level_start_index,
                           src_padding_mask, box_src_padding_mask=None, self_attn_mask=attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    # new_reference_points = reference_points
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args["hidden_dim"],
        nhead=args["nhead"],
        num_encoder_layers=args["enc_layers"],
        num_decoder_layers=args["dec_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"],
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args["num_feature_levels"],
        dec_n_points=args["dec_n_points"],
        enc_n_points=args["enc_n_points"],
        two_stage=args["two_stage"],
        two_stage_num_proposals=args["num_queries"],
        use_dab=True,
        num_classes=args["num_classes"]
    )


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    hidden_dim = 64
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
