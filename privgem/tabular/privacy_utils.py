#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is based on (with some minor changes):
# https://github.com/opendp/smartnoise-sdk
# 
# The above repo is released under MIT License.
# 
# Copy of the license on https://github.com/opendp/smartnoise-sdk:
# 
# MIT License
# 
# Copyright (c) 2020-2021 President and Fellows of Harvard College
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import math
import numpy as np


def weights_init(m):
    """
    Credit: This code is based on (with some minor changes):
            https://github.com/opendp/smartnoise-sdk
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def pate(data, teachers, lap_scale, device="cpu"):
    """
    Credit: This code is based on (with some minor changes):
            https://github.com/opendp/smartnoise-sdk
    """

    num_teachers = len(teachers)
    labels = torch.Tensor(num_teachers, data.shape[0]).type(torch.int64).to(device)
    for i in range(num_teachers):
        output = teachers[i](data)
        pred = (output > 0.5).type(torch.Tensor).squeeze().to(device)
        labels[i] = pred

    votes = torch.sum(labels, dim=0).unsqueeze(1).type(torch.DoubleTensor).to(device)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1 / lap_scale, size=votes.size())).to(
        device
    )
    noisy_votes = votes + noise
    noisy_labels = (noisy_votes > num_teachers / 2).type(torch.DoubleTensor).to(device)

    return noisy_labels, votes


def moments_acc(num_teachers, votes, lap_scale, l_list, device="cpu"):
    """
    Credit: This code is based on (with some minor changes):
            https://github.com/opendp/smartnoise-sdk
    """

    q = (2 + lap_scale * torch.abs(2 * votes - num_teachers)) / (
        4 * torch.exp(lap_scale * torch.abs(2 * votes - num_teachers))
    ).to(device)

    alpha = []
    for l_val in l_list:
        a = 2 * lap_scale ** 2 * l_val * (l_val + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l_val)
        t_two = q * torch.exp(2 * lap_scale * l_val)
        t = t_one + t_two
        alpha.append(torch.clamp(t, max=a).sum())

    return torch.DoubleTensor(alpha).to(device)
