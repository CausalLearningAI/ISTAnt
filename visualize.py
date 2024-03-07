import matplotlib.pyplot as plt
import os

def plot_outcome_distribution(data, save=False):
    T = data['treatment']
    Y = data['outcome']
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['yellow', 'blue']
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
        axs[i].set_title(f'Grooming {color}')
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        fig.savefig('results/outcome_distribution.png')
    else:
        plt.show();