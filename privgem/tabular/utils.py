#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

def split_save_orig_data(input_data,
                         path_save="orig_data.csv",
                         path_train=None,
                         path_test=None,
                         label_col=None,
                         test_size=0.25,
                         random_state=42):
    """
    Examples:
    path_save="./test/orig_data/orig_data.csv",
    path_train="./test/orig_data/orig_train.csv",
    path_test="./test/orig_data/orig_test.csv",
    """

    path_save = os.path.abspath(path_save)
    path2orig_dir = os.path.dirname(path_save)
    
    os.makedirs(path2orig_dir, exist_ok=True)

    # save the input data
    print(f"[INFO] save the original file: {path_save}")
    input_data.to_csv(path_save, index=False)

    print(f"[INFO] split dataset")
    if (path_train is not None) or (path_test is not None):
        data_train, data_test = train_test_split(input_data, 
                                                test_size=test_size, 
                                                random_state=random_state, 
                                                stratify=input_data[label_col] if label_col else None)
        
        print(f"[INFO] save the train set: {path_train}")
        data_train.to_csv(path_train, index=False)

        print(f"[INFO] save the test set: {path_test}")
        data_test.to_csv(path_test, index=False)

def plot_log_patectgan(filename, method_name="PATE-CTGAN", show_or_save="show"):

    with open(filename, "r") as fio:
        lines = fio.readlines()

    # split title and the body
    title = lines[0]
    title = title.split(",")[1:]
    title = ",".join(title)
    title = method_name + "\n" + title

    body = lines[1:]

    # number of iterations
    iters = [int(x.split("|")[0]) for x in body]
    # time 
    times = [float(x.split("|")[1]) for x in body]
    # eps, G and D losses
    eps = [float(x.split("|")[2].split("(")[0].split("eps:")[1]) for x in body]
    G_loss = [float(x.split("|")[3].split("G: ")[1]) for x in body]
    D_loss = [float(x.split("|")[4].split("D: ")[1]) for x in body]
    # accuracies
    acc_fake = [float(x.split("|")[5].split("Acc (fake): ")[1]) for x in body]
    acc_real = [float(x.split("|")[6].split("Acc (true): ")[1]) for x in body]
    acc_gen = [float(x.split("|")[7].split("Acc (generator): ")[1]) for x in body]
    acc_student = [float(x.split("|")[8].split("Acc (student): ")[1]) for x in body]

    # add max iter and eps to title
    title += f"max iter: {iters[-1]}, eps: {eps[-1]}"
    print(f"---\n{title}\n---")

    # --- params for the plots
    figsize = (15, 10)
    # number of subplots in y and x
    y_subs = 3
    x_subs = 2

    title_font_size = 18
    legend_fontsize = 14
    xlabel_font_size = 18
    ylabel_font_size = 18
    xticks_size = 14
    yticks_size = 14
    line_width = 3

    # --- FIGURE
    plt.figure(figsize=figsize)
    plt.suptitle(title, size=title_font_size)

    # --- subplot 1
    counter = 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, eps, 
             color="k", 
             lw=line_width)
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("$\epsilon$ (cumulative)", 
               size=ylabel_font_size)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 2
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters[1:], np.array(eps)[1:] - np.array(eps)[:-1], 
             color="k", 
             lw=line_width)
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("$\epsilon$ per iter", 
               size=ylabel_font_size)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 3
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, G_loss, 
             color="k", 
             lw=line_width, 
             label="loss (G)")
    plt.plot(iters, D_loss, 
             color="r", 
             lw=line_width, 
             label="loss (D)")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("loss", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 4
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, times, 
             color="k", 
             lw=line_width,
             label=f"time/iter: {times[-1] / iters[-1]:.3f}s")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("time", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 5
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, acc_fake, 
             color="k", 
             lw=line_width, 
             label="Acc (fake) - teacher")
    plt.plot(iters, acc_real, 
             color="r", 
             lw=line_width, 
             label="Acc (true) - teacher")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("Accuracy", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    #plt.legend(bbox_to_anchor=(0., -0.5, 1., -0.5), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 6
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, acc_gen, 
             color="y", 
             lw=line_width,
             label="Acc - generator")
    plt.plot(iters, acc_student, 
             color="g", 
             lw=line_width,
             label="Acc - student")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("Accuracy", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    #plt.legend(bbox_to_anchor=(0., -0.5, 1., -0.5), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    plt.tight_layout()
    if show_or_save in ["save"]:
        plt.savefig(f"eps_{eps[-1]}_{filename.split('.csv')[0]}.png")
    elif show_or_save in ["show"]:
        plt.show()

def plot_log_dpctgan(filename, method_name="DP-CTGAN", show_or_save="show"):
    
    with open(filename, "r") as fio:
        lines = fio.readlines()

    # split title and the body
    title = lines[0]
    title = title.split(",")[1:]
    title = ",".join(title)
    title = method_name + "\n" + title

    body = lines[1:]

    # number of iterations
    iters = [int(x.split("|")[0]) for x in body]
    # time 
    times = [float(x.split("|")[1]) for x in body]
    # eps, G and D losses
    eps = [float(x.split("|")[2].split("(")[0].split("eps:")[1]) for x in body]
    G_loss = [float(x.split("|")[3].split("G: ")[1]) for x in body]
    D_loss = [float(x.split("|")[4].split("D: ")[1]) for x in body]
    # accuracies
    acc_fake = [float(x.split("|")[5].split("Acc (fake): ")[1]) for x in body]
    acc_real = [float(x.split("|")[6].split("Acc (true): ")[1]) for x in body]
    acc_gen = [float(x.split("|")[7].split("Acc (generator): ")[1]) for x in body]
    alpha = [float(x.split("|")[8].split("alpha: ")[1]) for x in body]

    # add max iter and eps to title
    title += f"max iter: {iters[-1]}, eps: {eps[-1]}"
    print(f"---\n{title}\n---")

    # --- params for the plots
    figsize = (15, 10)
    # number of subplots in y and x
    y_subs = 3
    x_subs = 2

    title_font_size = 18
    legend_fontsize = 14
    xlabel_font_size = 18
    ylabel_font_size = 18
    xticks_size = 14
    yticks_size = 14
    line_width = 3

    # --- FIGURE
    plt.figure(figsize=figsize)
    plt.suptitle(title, size=title_font_size)

    # --- subplot 1
    counter = 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, eps, 
             color="k", 
             lw=line_width)
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("$\epsilon$ (cumulative)", 
               size=ylabel_font_size)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 2
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters[1:], np.array(eps)[1:] - np.array(eps)[:-1], 
             color="k", 
             lw=line_width)
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("$\epsilon$ per iter", 
               size=ylabel_font_size)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 3
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, G_loss, 
             color="k", 
             lw=line_width, 
             label="loss (G)")
    plt.plot(iters, D_loss, 
             color="r", 
             lw=line_width, 
             label="loss (D)")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("loss", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 4
    print(f"time/iter: {times[-1] / iters[-1]:.3f}s")
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, alpha, 
             color="k", 
             lw=line_width)
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("$\\alpha$", 
               size=ylabel_font_size)

    ### plt.plot(iters, times, 
    ###          color="k", 
    ###          lw=line_width,
    ###          label=f"time/iter: {times[-1] / iters[-1]:.3f}s")
    ### plt.xlabel("iters", 
    ###            size=xlabel_font_size)
    ### plt.ylabel("time", 
    ###            size=ylabel_font_size)
    ### plt.legend(fontsize=legend_fontsize)

    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 5
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, acc_fake, 
             color="k", 
             lw=line_width, 
             label="Acc (fake) - disc.")
    plt.plot(iters, acc_real, 
             color="r", 
             lw=line_width, 
             label="Acc (true) - disc.")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("Accuracy", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    #plt.legend(bbox_to_anchor=(0., -0.5, 1., -0.5), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    # --- subplot 6
    counter += 1
    plt.subplot(y_subs, x_subs, counter)
    plt.plot(iters, acc_gen, 
             color="y", 
             lw=line_width,
             label="Acc - generator")
    plt.xlabel("iters", 
               size=xlabel_font_size)
    plt.ylabel("Accuracy", 
               size=ylabel_font_size)
    plt.legend(fontsize=legend_fontsize)
    #plt.legend(bbox_to_anchor=(0., -0.5, 1., -0.5), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)
    plt.grid()

    plt.tight_layout()
    if show_or_save in ["save"]:
        plt.savefig(f"eps_{eps[-1]}_{filename.split('.csv')[0]}.png")
    elif show_or_save in ["show"]:
        plt.show()