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

import numpy as np
import os
import time

from sklearn.metrics import accuracy_score

import torch
from torch import optim
from torch import nn
import torch.utils.data
from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential, Sigmoid

from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator
from ctgan.models import Generator
from ctgan.sampler import Sampler
from ctgan import CTGANSynthesizer

import opacus


class Discriminator(Module):
    """
    Credit: This code is based on (with some minor changes):
            https://github.com/opendp/smartnoise-sdk
    """

    def __init__(self, input_dim, dis_dims, loss, pack):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pack
        #  print ('now dim is {}'.format(dim))
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == "cross_entropy":
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device="cpu", pac=10, lambda_=10):

        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


# custom for calcuate grad_sample for multiple loss.backward()
def _custom_create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or accumulate it
    if the 'grad_sample' attribute already exists.
    This custom code will not work when using optimizer.virtual_step()
    """

    # print ("now this happen")

    if hasattr(param, "grad_sample"):
        param.grad_sample = param.grad_sample + grad_sample
        # param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample


class dpctgan(CTGANSynthesizer):
    """Differential Private Conditional Table GAN Synthesizer
    This code adds Differential Privacy to CTGANSynthesizer from https://github.com/sdv-dev/CTGAN

    Credit: This code is based on (with some minor changes):
        https://github.com/opendp/smartnoise-sd
    """

    def __init__(
        self,
        embedding_dim=128,
        gen_dim=(256, 256),
        dis_dim=(256, 256),
        l2scale=1e-6,
        batch_size=500,
        epochs=300,
        pack=1,
        log_frequency=True,
        disabled_dp=False,
        target_delta=None,
        sigma=5,
        max_per_sample_grad_norm=1.0,
        epsilon=1,
        verbose=True,
        loss="cross_entropy",
        secure_rng=True,
        output_save_path="default.csv",
        device="default"
    ):

        # CTGAN model specific parameters
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.pack = pack
        self.log_frequency = log_frequency

        if device in ["default"]:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # opacus parameters
        self.sigma = sigma
        self.disabled_dp = disabled_dp
        self.target_delta = target_delta
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.epsilon = epsilon
        # self.epsilon_list = []
        # self.alpha_list = []
        # self.loss_d_list = []
        # self.loss_g_list = []
        self.verbose = verbose
        self.loss = loss
        self.secure_rng = secure_rng
        self.output_save_path = output_save_path

        if self.loss != "cross_entropy":
            # Monkeypatches the _create_or_extend_grad_sample function when calling opacus
            opacus.supported_layers_grad_samplers._create_or_extend_grad_sample = (
                _custom_create_or_extend_grad_sample
            )
            # XXX

        os.makedirs(os.path.dirname(os.path.abspath(output_save_path)), exist_ok=True)
        with open(f"{output_save_path}", "w") as fio:
            fio.writelines(f"DP-CTGAN, epsilon: {epsilon}, "
                           f"sigma: {sigma}, "
                           f"batch_size: {batch_size}\n")

    def train(self, data, categorical_columns=None, ordinal_columns=None, update_epsilon=None):
        if update_epsilon:
            self.epsilon = update_epsilon

        self.transformer = DataTransformer()
        self.transformer.fit(data, discrete_columns=categorical_columns)
        train_data = self.transformer.transform(data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data, self.transformer.output_info, self.log_frequency
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim
        ).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt, self.dis_dim, self.loss, self.pack
        ).to(self.device)

        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale
        )
        optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        privacy_engine = opacus.PrivacyEngine(
            discriminator,
            batch_size=self.batch_size,
            sample_size=train_data.shape[0],
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_per_sample_grad_norm,
            clip_per_layer=True,
            secure_rng=self.secure_rng
        )

        if not self.disabled_dp:
            privacy_engine.attach(optimizer_d)

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1

        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        counter = 0 
        t1 = time.time()
        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            # XXX
            all_fake_label = []; all_fake_y = []
            all_true_label = []; all_true_y = []
            all_gen_label = []; all_gen_y = []

            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype("float32")).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                optimizer_d.zero_grad()

                if self.loss == "cross_entropy":
                    y_fake = discriminator(fake_cat)

                    #   print ('y_fake is {}'.format(y_fake))
                    label_fake = torch.full(
                        (int(self.batch_size / self.pack),),
                        fake_label,
                        dtype=torch.float,
                        device=self.device,
                    )

                    #    print ('label_fake is {}'.format(label_fake))

                    error_d_fake = criterion(y_fake.squeeze(), label_fake)
                    error_d_fake.backward()
                    optimizer_d.step()

                    # train with real
                    label_true = torch.full(
                        (int(self.batch_size / self.pack),),
                        real_label,
                        dtype=torch.float,
                        device=self.device,
                    )
                    y_real = discriminator(real_cat)
                    error_d_real = criterion(y_real.squeeze(), label_true)
                    error_d_real.backward()
                    optimizer_d.step()

                    loss_d = error_d_real + error_d_fake

                    # XXX
                    all_fake_y.extend(y_fake.flatten().detach().cpu())
                    all_true_y.extend(y_real.flatten().detach().cpu())
                    all_fake_label.extend(label_fake.flatten().detach().cpu())
                    all_true_label.extend(label_true.flatten().detach().cpu())

                else:

                    y_fake = discriminator(fake_cat)
                    mean_fake = torch.mean(y_fake)
                    mean_fake.backward(one)

                    y_real = discriminator(real_cat)
                    mean_real = torch.mean(y_real)
                    mean_real.backward(mone)

                    optimizer_d.step()

                    loss_d = -(mean_real - mean_fake)

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                # pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                # pen.backward(retain_graph=True)
                # loss_d.backward()
                # optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # if condvec is None:
                cross_entropy = 0
                # else:
                #    cross_entropy = self._cond_loss(fake, c1, m1)

                if self.loss == "cross_entropy":
                    label_g = torch.full(
                        (int(self.batch_size / self.pack),),
                        real_label,
                        dtype=torch.float,
                        device=self.device,
                    )
                    # label_g = torch.full(int(self.batch_size/self.pack,),1,device=self.device)
                    loss_g = criterion(y_fake.squeeze(), label_g)
                    loss_g = loss_g + cross_entropy

                    # XXX
                    all_gen_y.extend(y_fake.flatten().detach().cpu())
                    all_gen_label.extend(label_g.float().detach().cpu())
                else:
                    loss_g = -torch.mean(y_fake) + cross_entropy

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                if not self.disabled_dp:
                    # if self.loss == 'cross_entropy':
                    #    autograd_grad_sample.clear_backprops(discriminator)
                    # else:
                    for p in discriminator.parameters():
                        if hasattr(p, "grad_sample"):
                            del p.grad_sample

                    if self.target_delta is None:
                        self.target_delta = 1 / train_data.shape[0]

                    epsilon, best_alpha = optimizer_d.privacy_engine.get_privacy_spent(
                        self.target_delta
                    )

                    # self.epsilon_list.append(epsilon)
                    # self.alpha_list.append(best_alpha)

            if not self.disabled_dp:
                if self.epsilon < epsilon:
                    break

            #self.loss_d_list.append(loss_d)
            #self.loss_g_list.append(loss_g)

            # XXX
            if self.verbose:
                outmsg = f"{counter} | {time.time() - t1:f} | "\
                         f"eps: {epsilon:.6f} (target: {self.epsilon:.6f}) | "\
                         f"G: {loss_g.detach().cpu():.4f} | D: {loss_d.detach().cpu():.4f} | "\
                         f"Acc (fake): {accuracy_score(all_fake_label, np.round(all_fake_y)):.4f} | "\
                         f"Acc (true): {accuracy_score(all_true_label, np.round(all_true_y)):.4f} | "\
                         f"Acc (generator): {accuracy_score(all_gen_label, np.round(all_gen_y)):.4f} | "\
                         f"alpha: {best_alpha}"\

                with open(self.output_save_path, "a+") as fio:
                    fio.writelines(outmsg + "\n")
                print(outmsg)
                counter += 1

        # return self.loss_d_list, self.loss_g_list, self.epsilon_list, self.alpha_list

    def generate(self, n):
        self.generator.eval()

        # output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)
