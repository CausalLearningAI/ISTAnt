import matplotlib.pyplot as plt
import torch
from data import get_examples
import os

def visualize_examples(n, model_name, model, outcome="all", environment="supervised", save=True, data_dir="./data", results_dir="./results"):
    if n < 6:
        columns = n
    else:
        columns = 6   
    rows = n//6 + 1
    fig = plt.figure(figsize=(15, rows*2.7))
    ax = []
    imgs, ys, embs = get_examples(environment=environment, 
                                  n=n, 
                                  outcome=outcome, 
                                  model_name=model_name,
                                  data_dir=data_dir)
    if environment=="supervised":
        for i, (img, y, emb) in enumerate(zip(imgs, ys, embs)):
            y_pred = [int(elem.item()) for elem in model.pred(emb)]
            y = [int(elem.item()) for elem in y]
            plt.rc('font', size=8)
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(f"H: {y}, ML: {y_pred}")
            plt.imshow(img.permute(1, 2, 0))
    elif environment=="unsupervised":
        for i, (img, emb) in enumerate(zip(imgs, embs)):
            y_pred = [int(elem.item()) for elem in model.pred(emb)]
            plt.rc('font', size=8)
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(f"ML: {y_pred}")
            plt.imshow(img.permute(1, 2, 0))
    else:
        raise ValueError(f"Environment {environment} not defined.")
    if save: 
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        title = f"example_{environment}_predictions.png"
        path_fig = os.path.join(results_dir, title)
        plt.savefig(path_fig, bbox_inches='tight')
    else:
        plt.show()

def plot_outcome_distribution(data, save=False, total=True, results_dir="./results"):
    T = data['treatment']
    Y = data['outcome']
    fig, axs = plt.subplots(1, 2+total, figsize=(12+6*total, 5))
    colors = ['Y2F', 'B2F']
    for i, color in enumerate(colors):
        p_Y_0_T_0 = (Y[T==0, i]==0).sum() / Y.shape[0]
        p_Y_1_T_0 = (Y[T==0, i]==1).sum() / Y.shape[0]
        p_Y_0_T_1 = (Y[T==1, i]==0).sum() / Y.shape[0]
        p_Y_1_T_1 = (Y[T==1, i]==1).sum() / Y.shape[0]
        p_Y_0_T_2 = (Y[T==2, i]==0).sum() / Y.shape[0]
        p_Y_1_T_2 = (Y[T==2, i]==1).sum() / Y.shape[0]
        axs[i].bar(['0', '1'], [p_Y_0_T_0, p_Y_1_T_0], alpha=0.75, label='T=0 (control)')
        axs[i].bar(['0', '1'], [p_Y_0_T_1, p_Y_1_T_1], bottom=[p_Y_0_T_0, p_Y_1_T_0], alpha=0.75, label='T=1 (beads)')
        axs[i].bar(['0', '1'], [p_Y_0_T_2, p_Y_1_T_2], bottom=[p_Y_0_T_0+p_Y_0_T_1, p_Y_1_T_0+p_Y_1_T_1], alpha=0.75, label='T=2 (infused beads)')
        axs[i].set_xlabel('Y')
        axs[i].set_ylabel('p(Y)')
        axs[i].set_ylim(0, 1)
        axs[i].legend()
        axs[i].set_title(f'Grooming ({color})')
    if total:
        Y_tot = Y.sum(axis=1)
        p_Y_0_T_0 = (Y_tot[T==0]==0).sum() / Y.shape[0]
        p_Y_1_T_0 = (Y_tot[T==0]==1).sum() / Y.shape[0]
        p_Y_2_T_0 = (Y_tot[T==0]==2).sum() / Y.shape[0]
        p_Y_0_T_1 = (Y_tot[T==1]==0).sum() / Y.shape[0]
        p_Y_1_T_1 = (Y_tot[T==1]==1).sum() / Y.shape[0]
        p_Y_2_T_1 = (Y_tot[T==1]==2).sum() / Y.shape[0]
        p_Y_0_T_2 = (Y_tot[T==2]==0).sum() / Y.shape[0]
        p_Y_1_T_2 = (Y_tot[T==2]==1).sum() / Y.shape[0]
        p_Y_2_T_2 = (Y_tot[T==2]==2).sum() / Y.shape[0]
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_0, p_Y_1_T_0, p_Y_2_T_0], alpha=0.75, label='T=0 (control)')
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_1, p_Y_1_T_1, p_Y_2_T_1], bottom=[p_Y_0_T_0, p_Y_1_T_0, p_Y_2_T_0], alpha=0.75, label='T=1 (beads)')
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_2, p_Y_1_T_2, p_Y_2_T_2], bottom=[p_Y_0_T_0+p_Y_0_T_1, p_Y_1_T_0+p_Y_1_T_1, p_Y_2_T_0+p_Y_2_T_1], alpha=0.75, label='T=2 (infused beads)')
        axs[2].set_xlabel('Y')
        axs[2].set_ylabel('p(Y)')
        axs[2].set_ylim(0, 1)
        axs[2].legend()
        axs[2].set_title(f'Grooming (total)')
    if save:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        title = "outcome_distribution.png"
        path_fig = os.path.join(results_dir, title)
        fig.savefig(path_fig)
    else:
        plt.show();