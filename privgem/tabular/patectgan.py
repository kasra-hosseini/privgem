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

import joblib
import math
import numpy as np
import os
import time

from sklearn.metrics import accuracy_score

import torch
from torch import nn

from torch import optim
import torch.utils.data
from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential, Sigmoid
from torch.autograd import Variable

from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator
from ctgan.models import Generator
from ctgan.sampler import Sampler
from ctgan import CTGANSynthesizer

from .privacy_utils import weights_init, pate, moments_acc


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

    def dragan_penalty(self, real_data, device="cpu", c=10, lambda_=10):
        alpha = torch.rand(real_data.shape[0], 1, device=device).expand(real_data.shape)
        delta = torch.normal(
            mean=0.0, std=c, size=real_data.shape, device=device
        )  # 0.5 * real_data.std() * torch.rand(real_data.shape)
        x_hat = Variable(alpha * real_data + (1 - alpha) * (real_data + delta), requires_grad=True)

        pred_hat = self(x_hat.float())

        gradients = torch.autograd.grad(
            outputs=pred_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(pred_hat.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        dragan_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return dragan_penalty

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class patectgan(CTGANSynthesizer):
    """
    Credit: This code is based on (with some minor changes):
            https://github.com/opendp/smartnoise-sdk
    """

    def __init__(
        self,
        embedding_dim=128,
        gen_dim=(256, 256),
        dis_dim=(256, 256),
        l2scale=1e-6,
        epochs=300,
        pack=1,
        log_frequency=True,
        disabled_dp=False,
        target_delta=None,
        sigma=5,
        max_per_sample_grad_norm=1.0,
        verbose=False,
        loss="cross_entropy",  # losses supported: 'cross_entropy', 'wasserstein'
        regularization=None,  # regularizations supported: 'dragan'
        binary=False,
        batch_size=500,
        teacher_iters=5,
        student_iters=5,
        sample_per_teacher=1000,
        epsilon=8.0,
        delta=1e-5,
        noise_multiplier=1e-3,
        moments_order=100,
        # XXX
        output_save_path="default.csv",
        device="default"
    ):

        # CTGAN model specifi3c parameters
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

        self.verbose = verbose
        self.loss = loss
        self.regularization = regularization if self.loss != "wasserstein" else "dragan"
        self.sample_per_teacher = sample_per_teacher
        self.noise_multiplier = noise_multiplier
        self.moments_order = moments_order
        self.output_save_path = output_save_path

        self.binary = binary
        self.batch_size = batch_size
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.epsilon = epsilon
        self.cur_eps = None
        self.delta = delta
        self.pd_cols = None
        self.pd_index = None
        self.train_counter = None

        # XXX
        os.makedirs(os.path.dirname(os.path.abspath(output_save_path)), exist_ok=True)
        with open(f"{output_save_path}", "w") as fio:
            fio.writelines(f"PATE-CTGAN, epsilon: {epsilon}, "
                           f"noise_multiplier: {noise_multiplier}, "
                           f"moments order: {moments_order}, "
                           f"batch_size: {batch_size}\n")

    def train(self, data, categorical_columns=None, ordinal_columns=None, update_epsilon=None):
        if update_epsilon:
            self.epsilon = update_epsilon

        sample_per_teacher = (
            self.sample_per_teacher if self.sample_per_teacher < len(data) else 1000
        )
        self.num_teachers = int(len(data) / sample_per_teacher) + 1
        self.transformer = DataTransformer()
        self.transformer.fit(data, discrete_columns=categorical_columns)
        data = self.transformer.transform(data)
        data_partitions = np.array_split(data, self.num_teachers)

        data_dim = self.transformer.output_dimensions

        self.cond_generator = ConditionalGenerator(
            data, self.transformer.output_info, self.log_frequency
        )

        # create conditional generator for each teacher model
        cond_generator = [
            ConditionalGenerator(d, self.transformer.output_info, self.log_frequency)
            for d in data_partitions
        ]

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim
        ).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt, self.dis_dim, self.loss, self.pack
        ).to(self.device)

        student_disc = discriminator
        student_disc.apply(weights_init)

        teacher_disc = [discriminator for i in range(self.num_teachers)]
        for i in range(self.num_teachers):
            teacher_disc[i].apply(weights_init)

        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale
        )
        optimizer_s = optim.Adam(student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizer_t = [
            optim.Adam(teacher_disc[i].parameters(), lr=2e-4, betas=(0.5, 0.9))
            for i in range(self.num_teachers)
        ]

        noise_multiplier = self.noise_multiplier
        alphas = torch.tensor([0.0 for i in range(self.moments_order)], device=self.device)
        l_list = 1 + torch.tensor(range(self.moments_order), device=self.device)
    
        if self.cur_eps == None:
            self.cur_eps = 0

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        real_label = 1
        fake_label = 0

        criterion = nn.BCELoss() if (self.loss == "cross_entropy") else self.w_loss

        if self.verbose:
            print("using loss {} and regularization {}".format(self.loss, self.regularization))

        if self.train_counter == None:
            self.train_counter = 0

        t1 = time.time()
        while self.cur_eps < self.epsilon:
            # train teacher discriminators
            # XXX
            all_fake_label = []; all_fake_y = []
            all_true_label = []; all_true_y = []
            all_student_label = []; all_student_y = []
            all_gen_label = []; all_gen_y = []

            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):
                    partition_data = data_partitions[i]
                    data_sampler = Sampler(partition_data, self.transformer.output_info)
                    fakez = torch.normal(mean, std=std).to(self.device)

                    condvec = cond_generator[i].sample(self.batch_size)

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

                    optimizer_t[i].zero_grad()

                    y_all = torch.cat([teacher_disc[i](fake_cat), teacher_disc[i](real_cat)])
                    label_fake = torch.full(
                        (int(self.batch_size / self.pack), 1),
                        fake_label,
                        dtype=torch.float,
                        device=self.device,
                    )
                    label_true = torch.full(
                        (int(self.batch_size / self.pack), 1),
                        real_label,
                        dtype=torch.float,
                        device=self.device,
                    )
                    labels = torch.cat([label_fake, label_true])
                
                    error_d = criterion(y_all, labels)
                    error_d.backward()

                    # XXX
                    all_fake_y.extend(teacher_disc[i](fake_cat).flatten().detach().cpu())
                    all_true_y.extend(teacher_disc[i](real_cat).flatten().detach().cpu())
                    all_fake_label.extend(label_fake.flatten().detach().cpu())
                    all_true_label.extend(label_true.flatten().detach().cpu())

                    if self.regularization == "dragan":
                        pen = teacher_disc[i].dragan_penalty(real_cat, device=self.device)
                        pen.backward(retain_graph=True)

                    optimizer_t[i].step()

            # train student discriminator
            for t_3 in range(self.student_iters):
                data_sampler = Sampler(data, self.transformer.output_info)
                fakez = torch.normal(mean, std=std)

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

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                else:
                    fake_cat = fake

                fake_data = fake_cat
                predictions, votes = pate(
                    fake_data, teacher_disc, noise_multiplier, device=self.device
                )

                output = student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(
                    self.num_teachers, votes, noise_multiplier, l_list, device=self.device
                )

                # XXX
                all_student_y.extend(output.flatten().detach().cpu())
                all_student_label.extend(predictions.flatten().float().detach().cpu())

                loss_s = criterion(output, predictions.float().to(self.device))

                optimizer_s.zero_grad()
                loss_s.backward()

                if self.regularization == "dragan":
                    vals = torch.cat([predictions, fake_data], axis=1)
                    ordered = vals[vals[:, 0].sort()[1]]
                    data_list = torch.split(
                        ordered, predictions.shape[0] - int(predictions.sum().item())
                    )
                    synth_cat = torch.cat(data_list[1:], axis=0)[:, 1:]
                    pen = student_disc.dragan_penalty(synth_cat, device=self.device)
                    pen.backward(retain_graph=True)

                optimizer_s.step()

            # print ('iterator {i}, student discriminator loss is {j}'.format(i=t_3, j=loss_s))

            # train generator
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
                y_fake = student_disc(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = student_disc(fakeact)

            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = self._cond_loss(fake, c1, m1)

            if self.loss == "cross_entropy":
                label_g = torch.full(
                    (int(self.batch_size / self.pack), 1),
                    real_label,
                    dtype=torch.float,
                    device=self.device,
                )

                # XXX
                all_gen_y.extend(y_fake.flatten().detach().cpu())
                all_gen_label.extend(label_g.float().flatten().detach().cpu())

                loss_g = criterion(y_fake, label_g.float())
                loss_g = loss_g + cross_entropy
            else:
                loss_g = -torch.mean(y_fake) + cross_entropy

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            self.cur_eps = min((alphas - math.log(self.delta)) / l_list)
            
            # XXX
            if(self.verbose):
                outmsg = f"{self.train_counter} | {time.time() - t1:f} | "\
                         f"eps: {self.cur_eps:.6f} (target: {self.epsilon:.6f}) | "\
                         f"G: {loss_g.detach().cpu():.4f} | D: {loss_s.detach().cpu():.4f} | "\
                         f"Acc (fake): {accuracy_score(all_fake_label, np.round(all_fake_y)):.4f} | "\
                         f"Acc (true): {accuracy_score(all_true_label, np.round(all_true_y)):.4f} | "\
                         f"Acc (generator): {accuracy_score(all_gen_label, np.round(all_gen_y)):.4f} | "\
                         f"Acc (student): {accuracy_score(all_student_label, np.round(all_student_y)):.4f}"
                with open(self.output_save_path, "a+") as fio:
                    fio.writelines(outmsg + "\n")
                print(outmsg)
                self.train_counter += 1

    def w_loss(self, output, labels):
        vals = torch.cat([labels, output], axis=1)
        ordered = vals[vals[:, 0].sort()[1]]
        data_list = torch.split(ordered, labels.shape[0] - int(labels.sum().item()))
        fake_score = data_list[0][:, 1]
        true_score = torch.cat(data_list[1:], axis=0)[:, 1]
        w_loss = -(torch.mean(true_score) - torch.mean(fake_score))
        return w_loss

    def generate(self, n):
        self.generator.eval()

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

        generated_data = self.transformer.inverse_transform(data, None)

        return generated_data

    def save(self, save_path="default.obj", force=False):
        """Save object"""
        if os.path.isfile(save_path):
            if force:
                os.remove(save_path)
            else:
                raise FileExistsError(f"file already exists: {save_path}")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'wb') as myfile:
            #pickle.dump(self.__dict__, myfile)
            joblib.dump(self.__dict__, myfile)

    def load(self, load_path, remove_after_load=False):
        """load class"""
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"file not found: {load_path}")

        with open(load_path, 'rb') as myfile:
            #objPickle = pickle.load(myfile)
            objPickle = joblib.load(myfile)

        if remove_after_load:
            os.remove(load_path)

        self.__dict__ = objPickle 
