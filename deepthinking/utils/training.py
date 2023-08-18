""" training.py
    Utilities for training models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from dataclasses import dataclass
from random import randrange
import random
import typing

import torch
from icecream import ic
from tqdm import tqdm
import numpy as np
import torch.nn
from torch.utils.tensorboard.writer import SummaryWriter

from deepthinking.utils.testing import get_predicted


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"
    width: "typing.Any"

def get_grad_norm(net):
    total_norm = 0
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_grad_max(net):
    total_norm = 0
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        total_norm = max(total_norm, p.grad.detach().data.max())
    return total_norm

def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k


def train(net, trainloader, mode: str, train_setup: TrainingSetup, device: str, writer: SummaryWriter, epoch: int):
    if mode == "progressive":
        train_loss, acc = train_progressive(net, trainloader, train_setup, device, writer, epoch)
    elif mode == 'delta':
        train_loss, acc = train_delta(net, trainloader, train_setup, device, writer, epoch)
    elif mode == 'deltarand':
        train_loss, acc = train_delta(net, trainloader, train_setup, device, writer, epoch, rand=True)
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return train_loss, acc


def train_progressive(net, trainloader, train_setup: TrainingSetup, device: str, writer: SummaryWriter, epoch: int):
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    last_batch_idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        last_batch_idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        mask : typing.Any = (inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0) if problem == "mazes" else None

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, _ = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes":
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        loss.backward()
        global_step = epoch*len(trainloader)+batch_idx
        writer.add_scalar("batch_loss", loss, global_step)
        writer.add_scalar("grad_norm", get_grad_norm(net), global_step)
        writer.add_scalar("grad_max", get_grad_max(net), global_step)
        writer.add_scalar("batch_loss_last", loss_max_iters_mean, global_step)
        writer.add_scalar("batch_loss_progressive", loss_progressive_mean, global_step)
        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"batch_lr/group{i}",
                              optimizer.param_groups[i]["lr"],
                              global_step)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip) # type: ignore
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (last_batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc

def train_delta(net, trainloader, train_setup : TrainingSetup, device, writer : SummaryWriter, epoch : int, rand=False):
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    max_iters = train_setup.max_iters
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    last_batch_idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=True)):
        last_batch_idx = batch_idx
        assert max_iters >= 2, "need at least 2 iterations for delta training"
        batch_max_iter = random.randint(1, max_iters-1) if rand else max_iters-1
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        mask: typing.Any = (inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0) if problem == "mazes" else None

        optimizer.zero_grad()

        all_outputs = []
        interim_thought = None
        for i in range(max_iters):
            outputs, interim_thought = net(inputs, iters_to_do=1, interim_thought=interim_thought)
            all_outputs.append(outputs)

        all_losses = torch.zeros((max_iters)).to(inputs.device)
        for i, outputs in enumerate(all_outputs):
            nice_outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss = criterion(nice_outputs, targets)
            if problem == "mazes":
                loss = (loss * mask)
                loss = loss[mask > 0]
            all_losses[i] = loss.mean() # type: ignore

        delta_loss = all_losses[batch_max_iter] - all_losses[0]
        max_loss = all_losses[:batch_max_iter+1].max()
        max_loss_index = torch.argmax(all_losses).item()
        loss = delta_loss + max_loss
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")
        loss.backward()
        global_step = epoch*len(trainloader)+batch_idx
        writer.add_scalar("batch_loss", loss, global_step)
        writer.add_scalar("batch_loss_max", max_loss, global_step)
        writer.add_scalar("batch_loss_delta", delta_loss, global_step)
        writer.add_scalar("batch_loss_first", all_losses[0], global_step)
        writer.add_scalar("batch_loss_last", all_losses[-1], global_step)
        writer.add_scalar("batch_loss_max_index", max_loss_index, global_step)
        writer.add_scalar("grad_norm", get_grad_norm(net), global_step)
        writer.add_scalar("grad_max", get_grad_max(net), global_step)
        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"batch_lr/group{i}",
                              optimizer.param_groups[i]["lr"],
                              global_step)

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip) # type: ignore
        optimizer.step()

        train_loss += all_losses[-1].item()
        predicted = get_predicted(inputs, all_outputs[-1], problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)


    train_loss = train_loss / (last_batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc