# python imports
import argparse
import os
import time
import datetime
import glob
from pprint import pprint
from easydict import EasyDict

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from tensorboardX import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch_AF, test_one_epoch_AF, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
import json
import clip

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    # data loaders
    train_loader = make_data_loader(train_dataset, True, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, True, None, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    """ ActionFormer """
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    # enable model EMA
    model_ema = ModelEma(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/16", device=device)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_root_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_root_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_root_folder):
        os.mkdir(ckpt_root_folder)

    """4. pre-training loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

    ckpt_folder = os.path.join(ckpt_root_folder, "pretrain")
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    # tb_writer = None
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    is_best = False
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch_AF(
            train_loader,
            clip_model,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq)

        if (
                (epoch == max_epochs - 1) or
                is_best or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                is_best,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    if tb_writer is not None:
        tb_writer.close()

    """5. fine-tuning loop"""
    # test_cfg = load_config("configs/anet_i3d_AF.yaml")
    # test_cfg = load_config("configs/anet_slowfast_AF.yaml")
    # test_cfg = load_config("configs/anet_slowfast_TD.yaml")
    test_cfg = load_config("configs/thumos_i3d.yaml")
    # test_cfg = load_config("configs/thumos_slowfast_AF.yaml")
    # test_cfg = load_config("configs/thumos_slowfast_TD.yaml")
    test_dataset = make_dataset(test_cfg['dataset_name'], False, test_cfg['val_split'], **test_cfg['dataset'])
    # set bs = 1, and disable shuffle
    test_loader = make_data_loader(test_dataset, False, False, None, 1, cfg['loader']['num_workers'])

    # set up evaluator
    output_file = None
    test_db_vars = test_dataset.get_attributes()
    det_eval = ANETdetection(
        test_dataset.json_file,
        test_dataset.split[0],
        tiou_thresholds=test_db_vars['tiou_thresholds'],
    )

    cfg = test_cfg

    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    checkpoint = torch.load(os.path.join(ckpt_root_folder, "pretrain/epoch_009.pth.tar"))
    # checkpoint = torch.load(os.path.join("ckpt/kinetics_slowfast_TD_LTP_Re", "pretrain/epoch_009.pth.tar"))
    filtered_ckpt = dict()
    for k, v in checkpoint['state_dict_ema'].items():
        # if "query_embed" not in k:
        if "cls_head" not in k and "query_embed" not in k:
        # if "backbone.embd" not in k and "cls_head" not in k and "query_embed" not in k:
            filtered_ckpt[k] = v
    model.load_state_dict(filtered_ckpt, strict=False)
    del checkpoint

    model_ema = ModelEma(model)

    if args.percent < 1.0:
        cfg["dataset"]["data_percent"] = args.percent
    train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'])
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    # data loaders
    train_loader = make_data_loader(train_dataset, True, True, rng_generator, **cfg['loader'])

    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

    ckpt_folder = os.path.join(ckpt_root_folder, "finetune")
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    # tb_writer = None
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # mAP, APs, results = test_one_epoch(
    #     test_loader,
    #     clip_model,
    #     # detr,
    #     detr_model_ema.module,
    #     0,
    #     cfg['test_cfg'],
    #     evaluator=det_eval,
    #     output_file=output_file,
    #     ext_score_file=cfg['test_cfg']['ext_score_file'],
    #     tb_writer=tb_writer,
    #     print_freq=args.print_freq,
    #     output_dir=ckpt_folder
    # )
    # exit()

    is_best = False
    best_mAP = -1
    best_APs = None
    best_epoch = 0
    best_results = None
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch_AF(
            train_loader,
            clip_model,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq)

        if (epoch >= 0 and epoch % 1 == 0) or epoch == max_epochs - 1:
            mAP, APs, results = test_one_epoch_AF(
                test_loader,
                clip_model,
                model_ema.module,
                epoch,
                cfg['test_cfg'],
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writer,
                print_freq=args.print_freq,
                output_dir=ckpt_folder
            )

            is_best = mAP >= best_mAP
            if is_best:
                best_mAP = mAP
                best_APs = APs
                best_epoch = epoch
                best_results = results

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                is_best or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                is_best,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    # print the results
    result_txt = ""
    result_txt += '[RESULTS] Action detection results on {:s} at E{:02d}.'.format(cfg['dataset_name'],
                                                                                  best_epoch) + "\n"
    block = ''
    for tiou, tiou_mAP in zip(det_eval.tiou_thresholds, best_APs):
        block += '\n|tIoU = {:.2f}: mAP = {:.2f} (%)'.format(tiou, tiou_mAP * 100)
    result_txt += block + "\n"
    result_txt += 'Avearge mAP: {:.2f} (%)'.format(best_mAP * 100)
    print(result_txt)

    result_dict = dict({"version": "VERSION 1.3",
                        "results": dict(),
                        "external_data":
                            {"used": True,
                             "details": "3D-CNN for feature extracting is pre-trained on Kinetics-400"}})

    for r_i in range(len(best_results["video-id"])):
        video_id = best_results["video-id"][r_i]
        start_time = best_results["t-start"][r_i].item()
        end_time = best_results["t-end"][r_i].item()
        label = best_results["label"][r_i].item()
        score = best_results["score"][r_i].item()

        if video_id not in result_dict["results"].keys():
            result_dict["results"][video_id] = list()

        result_dict["results"][video_id].append({"label": label, "score": score, "segment": (start_time, end_time)})

    result_json_path = os.path.join(ckpt_folder, "results.json")
    with open(result_json_path, "w") as fp:
        json.dump(result_dict, fp, indent=4, sort_keys=True)

    result_text_path = os.path.join(ckpt_folder, "results.txt")
    with open(result_text_path, "w") as fp:
        fp.write(result_txt)

    # wrap up
    if tb_writer is not None:
        tb_writer.close()

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--percent', default=1.0, type=float)
    args = parser.parse_args()
    main(args)
