# python imports
import argparse
import os
import time
import datetime
import json
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


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

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

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
    train_loader = make_data_loader(
        train_dataset, True, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    checkpoint = torch.load(os.path.join(ckpt_root_folder, "pretrain/epoch_014.pth.tar"))
    # checkpoint = torch.load(os.path.join("ckpt/kinetics_slowfast_AF_base_origin_filtered", "pretrain/epoch_010.pth.tar"))
    filtered_ckpt = dict()
    for k, v in checkpoint['state_dict_ema'].items():
        # if "query_embed" not in k:
        if "cls_head.cls_head" not in k and "query_embed" not in k:
            filtered_ckpt[k] = v
    model.load_state_dict(filtered_ckpt, strict=False)
    del checkpoint

    model_ = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    model_ = nn.DataParallel(model_, device_ids=cfg['devices'])
    model_.load_state_dict(model.state_dict())
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model_, copy_model=False)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # set up evaluator
    output_file = None
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    is_best = False
    best_mAP = -1
    best_APs = None
    best_epoch = 0
    best_results = None
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )

        mAP, APs, results = valid_one_epoch(
            val_loader,
            model_ema.module,
            epoch,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            print_freq=args.print_freq)

        is_best = mAP >= best_mAP
        if is_best:
            best_mAP = mAP
            best_APs = APs
            best_epoch = epoch
            best_results = results

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
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
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
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

        result_dict["results"][video_id].append(
            {"label": label, "score": score, "segment": (start_time, end_time)})

    result_json_path = os.path.join(ckpt_folder, "results.json")
    with open(result_json_path, "w") as fp:
        json.dump(result_dict, fp, indent=4, sort_keys=True)

    result_text_path = os.path.join(ckpt_folder, "results.txt")
    with open(result_text_path, "w") as fp:
        fp.write(result_txt)

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
    args = parser.parse_args()
    main(args)
