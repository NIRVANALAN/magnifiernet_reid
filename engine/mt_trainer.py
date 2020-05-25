# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP

global ITER
ITER = 0

def create_mt_supervised_trainer(model, optimizer, loss_fn, weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, seg_target, attr_target, _, _ = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        if weight[2] != 0:
            attr_target = attr_target.to(device) if torch.cuda.device_count() >= 1 else attr_target
        seg_target = seg_target.to(device) if torch.cuda.device_count() >= 1 else seg_target
        
        output = model(x=img, seg_label=seg_target)
        
        if isinstance(output, list):
            id_output = output[0]
            seg_output = output[1]
            attr_output = output[2]
            mask_output = output[3]
            mask_beta = output[4]
            part_outputs = output[5]
            part_anchor_output = output[6]
            part_betas = output[7]
            part_schrodinger = output[8] #xuedinge
            glb_feat = output[9]
            mask_feat = output[10]
            part_div_feat = output[11]
        cri_id = loss_fn[0]
        cri_seg = loss_fn[1]
        cri_attr = loss_fn[2]
        cri_triplet = loss_fn[3]
        cri_ce = loss_fn[4] # a method to force intended branch to use CE loss
        cri_dv = loss_fn[5]

        loss_mask = torch.Tensor([0])
        loss_parts = [torch.Tensor([0])]
        loss_anchor = torch.Tensor([0])
        loss_triplet = torch.Tensor([0])
        loss_attr = torch.Tensor([0]).cuda()
        loss_dv = torch.Tensor([0])
        
        try:
            loss_cls = cri_id(id_output, target) #normal softmax
        except:
            loss_cls = cri_id(id_output, glb_feat, target) #softmax_triplet
        if mask_output is not None:
            try:
                loss_mask = cri_ce(mask_output, target)
            except:
                loss_mask = cri_id(mask_output, mask_feat, target)
        if part_outputs is not None:
            loss_parts = []
            for idx, p in enumerate(part_outputs):
                loss_parts.append(cri_ce(p, target))
            loss_anchor = cri_ce(part_anchor_output, target)
            try:
                loss_anchor = cri_id(part_anchor_output, target)
            except:
                loss_anchor = cri_id(part_anchor_output, part_schrodinger[1], target)
            loss_dv = cri_dv(part_schrodinger[2])
        loss_seg = cri_seg(seg_output, seg_target)
        # disable attr in a hard way, should change in future
        if weight[2] != 0:
            #print('Using attribute')
            loss_attr = cri_attr(attr_output, attr_target)
        loss = (weight[0] * loss_cls + weight[1] * loss_seg)
        if weight[2] != 0:
            loss += loss_attr
        if mask_output is not None:
            loss += weight[3] * loss_mask * mask_beta
        if part_outputs is not None:
            loss += weight[5] * loss_anchor
            for i, lp in enumerate(loss_parts):
                loss += weight[4] * part_betas[i] * lp
        #loss += weight[6] * loss_triplet # training for tripler losss is depreciated
        if weight[7] != 0:
            loss += weight[7] * loss_dv
        loss.backward()
        optimizer.step()
        
        # compute acc
        #acc = (score.max(1)[1] == target).float().mean() disabled
        return loss.item(), loss_seg.item(), loss_attr.item(), (sum(loss_parts)/len(loss_parts)).item(), \
                     loss_cls.item(), loss_anchor.item(), loss_mask.item(), loss_triplet.item(), \
                     loss_dv.item(), 0
    return Engine(_update)


def create_mt_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_mt_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        weight
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_mt.train")
    logger.info("Start training")
    trainer = create_mt_supervised_trainer(model, optimizer, loss_fn, weight, device=device)
    evaluator = create_mt_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=20, require_empty=False, save_as_state_dict=True)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_loss_seg')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_loss_attr')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'avg_loss_parts')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'avg_loss_glb')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'avg_loss_anchor')
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'avg_loss_mask')
    RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'avg_loss_triplet')
    RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'avg_loss_dv')
    RunningAverage(output_transform=lambda x: x[9]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Glb: {:.3f} Mask: {:.3f} \
               Anchor: {:.3f} Parts: {:.3f} Triplet: {:.3f} Seg: {:.3f} Attr: {:.3f} Div: {:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], \
                                engine.state.metrics['avg_loss_glb'], \
                                engine.state.metrics['avg_loss_mask'], \
                                engine.state.metrics['avg_loss_anchor'], \
                                engine.state.metrics['avg_loss_parts'], \
                                engine.state.metrics['avg_loss_triplet'], \
                                engine.state.metrics['avg_loss_seg'], \
                                engine.state.metrics['avg_loss_attr'], \
                                engine.state.metrics['avg_loss_dv'], \
                                engine.state.metrics['avg_acc'], \
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    print('Start iteration')
    trainer.run(train_loader, max_epochs=epochs)
