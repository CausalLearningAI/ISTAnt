import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import cv2
import os

from datasets import Dataset
from model import get_embeddings, MLP
from train import train_model
from visualize import plot_outcome_distribution
from utils import get_metric, set_seed, check_folder
from causal import compute_ate

class PPCI():
    def __init__(self, task="all", encoder="dino", token="class", split_criteria="experiment", reduce_fps_factor=15, downscale_factor=1, batch_size=100, num_proc=4, environment="all", generate=False, data_dir="./data/istant_lq", results_dir="./results/istant_lq", verbose=False):
        # TODO: fix generate option
        if environment in ["all", "supervised"]:
            self.supervised = load_env("supervised", 
                                    task=task, 
                                    encoder=encoder, 
                                    token=token, 
                                    split_criteria=split_criteria, 
                                    reduce_fps_factor=reduce_fps_factor, 
                                    downscale_factor=downscale_factor,
                                    batch_size=batch_size, 
                                    num_proc=num_proc,
                                    generate=generate,
                                    data_dir=data_dir,
                                    verbose=verbose)
            self.n_supervised = self.supervised["T"].shape[0]
        if environment in ["all", "unsupervised"]:
            self.unsupervised = load_env("unsupervised", 
                                    task=task, 
                                    encoder=encoder, 
                                    token=token, 
                                    split_criteria=split_criteria, 
                                    reduce_fps_factor=reduce_fps_factor, 
                                    downscale_factor=downscale_factor,
                                    batch_size=batch_size, 
                                    num_proc=num_proc,
                                    generate=generate,
                                    data_dir=data_dir,
                                    verbose=verbose)
            self.n_unsupervised = self.unsupervised["T"].shape[0]
        self.task = task
        self.encoder = encoder
        self.token = token
        self.split_criteria = split_criteria
        self.data_dir = data_dir
        self.results_dir = results_dir
        if verbose: print("Prediction-Powered Causal Inference dataset successfully loaded.")
    
    def train(self, batch_size=256, num_epochs=10, lr=0.001, hidden_nodes=256, hidden_layers=2, verbose=True, add_pred_env="supervised", seed=0, save=False, force=False):
        set_seed(seed)
        model_path = os.path.join(self.results_dir, "models", self.encoder, self.token, self.split_criteria, self.task, str(hidden_layers), str(lr), str(seed), "model.pth")
        if os.path.exists(model_path) and not force:
            if verbose: print("Model already trained.")
            self.model = MLP(self.supervised["X"].shape[1], hidden_nodes, hidden_layers, task=self.supervised["Y"].task)
            self.model.load_state_dict(torch.load(model_path))
            self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.model.device)
        else:
            self.model = train_model(self.supervised["X"], 
                                    self.supervised["Y"], 
                                    self.supervised["split"], 
                                    batch_size=batch_size, 
                                    num_epochs=num_epochs, 
                                    lr=lr, 
                                    hidden_nodes = hidden_nodes, 
                                    hidden_layers = hidden_layers,
                                    verbose=verbose)
            if save:
                model_dir = os.path.join(self.results_dir, "models", self.encoder, self.token, self.split_criteria, self.task, str(hidden_layers), str(lr), str(seed))
                check_folder(model_dir)
                torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pth"))
        if add_pred_env in ["supervised", "unsupervised"]:
            self.add_pred(add_pred_env)
        elif add_pred_env=="all":
            self.add_pred("supervised")
            self.add_pred("unsupervised")
        else:
            raise ValueError(f"Invalid add_pred_env argumen '{add_pred_env}', please select among: 'supervised', 'unsupervised', or 'all'.")
    
    def plot_out_distribution(self, save=True, total=True):
        if self.task=="all":
            plot_outcome_distribution(self.supervised, save=save, total=total, results_dir=self.results_dir)
        else:
            raise ValueError("Plot available only for task: 'all'.")

    def add_pred(self, environment="supervised"):
        if hasattr(self, 'model'):
            device = self.model.device
            with torch.no_grad():
                if environment=="supervised":
                    self.supervised["Y_hat"] = self.model.cond_exp(self.supervised["X"].to(device)).to("cpu").squeeze()
                elif environment=="unsupervised":
                    self.unsupervised["Y_hat"] = self.model.cond_exp(self.unsupervised["X"].to(device)).to("cpu").squeeze()
                elif environment=="all":
                    self.supervised["Y_hat"] = self.model.cond_exp(self.supervised["X"].to(device)).to("cpu").squeeze()
                    self.unsupervised["Y_hat"] = self.model.cond_exp(self.unsupervised["X"].to(device)).to("cpu").squeeze()
                else:
                    raise ValueError(f"Environment '{environment}' not defined.")
        else:
            raise ValueError("Train the model first, before computing the inference step.")
    
    def evaluate(self, color="blue", T_control=1, T_treatment=2, verbose=False):
        if "Y_hat" in self.supervised:
            if self.task=="all":
                if color=="yellow":
                    Y = self.supervised["Y"][:,0]
                    Y_hat = self.supervised["Y_hat"][:,0]
                elif color=="blue":
                    Y = self.supervised["Y"][:,1]
                    Y_hat = self.supervised["Y_hat"][:,1]
                else:
                    raise ValueError(f"Invalid color '{color}', please select between: 'blue', 'yellow'.")
            else:
                Y = self.supervised["Y"]
                Y_hat = self.supervised["Y_hat"]
            color = "preselected"
            W = self.supervised["W"]
            T = self.supervised["T"]
            split = self.supervised["split"]
            n_val = 1000
            idx = random.sample(range(0, (~split).sum()), n_val)
            #idx = list(range(0, n_val))
            Y_val = Y[~split][idx]
            Y_hat_val = Y_hat[~split][idx]
            T_val = T[~split][idx]
            W_val = W[~split][idx]
            # validation
            pos_wwight = ((Y[split]==0).sum(dim=0)/(Y[split]==1).sum(dim=0))#.to(device)
            loss_fn = torch.nn.BCELoss(weight=pos_wwight)
            loss_val = loss_fn(Y_hat_val, Y_val).item()
            acc_val = get_metric(Y_val, Y_hat_val.round(), metric="accuracy")
            bacc_val = get_metric(Y_val, Y_hat_val.round(), metric="balanced_acc")
            TEB_val = compute_ate(Y_hat_val,T_val, W_val, method="ead", color=color, T_control=T_control, T_treatment=T_treatment) - compute_ate(Y_val, T_val, W_val, method="ead", color=color, T_control=T_control, T_treatment=T_treatment)
            
            # all
            acc = get_metric(Y, Y_hat.round(), metric="accuracy")
            bacc = get_metric(Y, Y_hat.round(), metric="balanced_acc")
            EAD = compute_ate(Y, T, W, method="ead", color=color, T_control=T_control, T_treatment=T_treatment) 
            TEB = compute_ate(Y_hat,T, W, method="ead", color=color, T_control=T_control, T_treatment=T_treatment) - EAD
            TEB_bin = compute_ate(Y_hat.round(), T, W, method="ead", color=color, T_control=T_control, T_treatment=T_treatment) - EAD
 
            metric = {
                "loss_val": loss_val,
                "acc_val": acc_val,
                "bacc_val": bacc_val,
                "TEB_val": TEB_val,
                "acc": acc,
                "bacc": bacc,
                "TEB": TEB,
                "TEB_bin": TEB_bin,
                "EAD": EAD,
            }
            if verbose: print(metric)
            return metric
        else:
            raise ValueError("Train the model and predict the labels on the supervised dataset, before measuring the performances.")

    def get_examples(self, n, environment="supervised", validation=False):
        if environment=="supervised":
            if validation:
                val_indeces = torch.nonzero(~self.supervised["split"]).squeeze()
                idxs = random.sample(val_indeces.tolist(), n)
            else:
                train_indeces = torch.nonzero(self.supervised["split"]).squeeze()
                idxs = random.sample(train_indeces.tolist(), n)
            exps = self.supervised["source_data"][idxs]["experiment"]
            poss = self.supervised["source_data"][idxs]["position"]
            exp = [chr(97+exp)+str(pos.item()) for exp, pos in zip(exps, poss)]
            frame = self.supervised["source_data"][idxs]["frame"]
            image = self.supervised["source_data"][idxs]["image"]
            Y = self.supervised["Y"][idxs] 
            if "Y_hat" in self.supervised:
                Y_hat = self.supervised["Y_hat"][idxs] 
            else:
                Y_hat = None
        elif environment=="unsupervised":
            idxs = torch.randint(0, self.n_unsupervised, (n,))
            image = self.unsupervised["source_data"][idxs]["image"]
            exps = self.supervised["source_data"][idxs]["experiment"]
            poss = self.supervised["source_data"][idxs]["position"]
            exp = [chr(97+exp)+str(pos.item()) for exp, pos in zip(exps, poss)]
            frame = self.unsupervised["source_data"][idxs]["frame"]
            Y = None
            if "Y_hat" in self.unsupervised:
                Y_hat = self.unsupervised["Y_hat"][idxs] 
            else:
                Y_hat = None
        else:
            raise ValueError(f"Environemnt '{environment}' not defined, please select between: 'supervised' and 'unsupervised'.")
        return image, Y, Y_hat, exp, frame

    def visualize(self, save=True, k=6, detailed=True):
        train, test = False, False
        if hasattr(self, 'supervised'):
            if ("Y_hat" in self.supervised):
                train = True
        if hasattr(self, 'unsupervised'):
            if ("Y_hat" in self.unsupervised):
                test = True
        if train+test==0:
            raise ValueError("Generate first at least an environment, train a model and predict the corresponding labels before visualizing.")
        fig = plt.figure(figsize=(k*2.5, 0.4+4.4*train+2.2*test))
        ax = []
        if train: 
            # train
            images, Ys, Y_hats, exps, frames = self.get_examples(k, environment="supervised", validation=False)
            for i, (img, y, y_pred, exp, frame) in enumerate(zip(images, Ys, Y_hats.round(), exps, frames)):
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                y = [int(elem.item()) for elem in y.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + 1))
                if detailed: title = f"H: {y}, ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"H: {y}, ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[0].annotate('Training', xy=(0, 0.5), xytext=(-ax[0].yaxis.labelpad - 5, 0),
                            xycoords=ax[0].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
            # validation
            images, Ys, Y_hats, exps, frames = self.get_examples(k, environment="supervised", validation=True)
            for i, (img, y, y_pred, exp, frame) in enumerate(zip(images, Ys, Y_hats.round(), exps, frames)):
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                y = [int(elem.item()) for elem in y.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + k + 1))
                if detailed: title = f"H: {y}, ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"H: {y}, ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[k].annotate('Validation', xy=(0, 0.5), xytext=(-ax[k].yaxis.labelpad - 5, 0),
                            xycoords=ax[k].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
        if test:
            # test
            images, _, Y_hats = self.get_examples(k, environment="unsupervised")
            for i, (img, y_pred, exp, frame) in enumerate(zip(images, Y_hats.round(), exps, frames)):
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + 2*train*k +1))
                if detailed: title = f"ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[2*train*k].annotate('Test', xy=(0, 0.5), xytext=(-ax[2*train*k].yaxis.labelpad - 5, 0),
                            xycoords=ax[2*train*k].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
        if save: 
            results_example_dir = os.path.join(self.results_dir, "example_pred")
            if not os.path.exists(results_example_dir):
                os.makedirs(results_example_dir)
            title = f"{self.encoder}_{self.token}_task_{self.task}.png"
            path_fig = os.path.join(results_example_dir, title)
            plt.savefig(path_fig, bbox_inches='tight')
        else:
            plt.show()

    def __str__(self):
        return "Prediction-Powered Causal Inference dataset (PPCI object)"

    def __repr__(self):
        return "Prediction-Powered Causal Inference dataset (PPCI object)"

def get_outcome(dataset, task="all"):
    if task=="all":
        y = dataset["outcome"]
    elif task.lower()=="yellow":
        y = dataset["outcome"][:,0]
    elif task.lower()=="blue":
        y = dataset["outcome"][:,1]
    elif task.lower()=="sum":
        y = dataset["outcome"].sum(axis=1)
    elif task.lower()=="or":
        y = torch.logical_or(dataset["outcome"][:,0], dataset["outcome"][:,1]).float()
    else:
        raise ValueError(f"Task {task} not defined. Please select between: 'all', 'yellow', 'blue', 'sum', 'or'.")
    y.task = task
    return y

def get_split(dataset, split_criteria="experiment"):
    if split_criteria=="experiment":
        split = (dataset["experiment"] == 0) # tr_ration: 1/5
    elif split_criteria=="experiment_easy":
        split = (dataset["experiment"] != 4) # tr_ration: 4/5
    elif split_criteria=="position":
        split = (dataset["position"] == 1) # tr_ration: 1/9
    elif split_criteria=="position_easy":
        split = (dataset["position"] != 8) # tr_ration: 8/9
    elif split_criteria=="random":
        split = torch.zeros_like(dataset["experiment"], dtype=torch.bool)
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        for exp_i, pos_i in zip(exps,poss):
            split_i = (dataset["experiment"]==exp_i) & (dataset["position"]==pos_i)
            split = split | split_i # tr_ration: 1/5
    elif split_criteria=="random_easy":
        split = torch.ones_like(dataset["experiment"], dtype=torch.bool)
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        for exp_i, pos_i in zip(exps,poss):
            split_i = (dataset["experiment"]!=exp_i) | (dataset["position"]!=pos_i)
            split = split & split_i # tr_ration: 4/5
    else:
        raise ValueError(f"Split criteria {split_criteria} doesn't exist. Please select a valid splitting criteria: 'experiment', 'position' and 'random'.")
    split.criteria = split_criteria
    return split

def get_covariates(dataset):
    covariates = ['pos_x', 'pos_y', 'exp_minute', 'experiment']
    W = torch.stack([dataset[covariate] for covariate in covariates[:-1]], dim=1)
    W_exp = torch.nn.functional.one_hot(dataset["experiment"], num_classes=len(dataset["experiment"].unique()))
    W = torch.cat([W, W_exp], dim=1)
    return W

def load_env(environment='supervised', task="all", encoder="mae", token="class", split_criteria="experiment", generate=False, reduce_fps_factor=10, downscale_factor=1, batch_size=100, num_proc=4, data_dir="./data", verbose=False):
    data_env_dir = os.path.join(data_dir, environment)
    if not os.path.exists(data_env_dir):
        os.makedirs(data_env_dir)
        generate = True
    if generate:
        dataset = Dataset.from_generator(generator, gen_kwargs={"reduce_fps_factor": reduce_fps_factor, "downscale_factor": downscale_factor, "environment":environment, "data_dir":data_dir})
        dataset.save_to_disk(data_env_dir)
        if verbose: print("Data generated and saved correctly.")
    else:
        dataset = Dataset.load_from_disk(data_env_dir)
    dataset.set_format(type="torch", columns=["image", "treatment", "outcome", 'pos_x', 'pos_y', 'exp_minute', 'day_hour', 'frame', "experiment", "position"], output_all_columns=True)
    dataset.environment = environment
    dataset_dict = {
        "source_data": dataset,
        "X": get_embeddings(dataset, encoder, batch_size=batch_size, num_proc=num_proc, data_dir=data_dir, token=token, verbose=verbose),
        "Y": get_outcome(dataset, task=task),
        "split": get_split(dataset, split_criteria=split_criteria),
        "W": get_covariates(dataset), 
        "T": dataset["treatment"],
    }
    return dataset_dict

def generator(reduce_fps_factor, downscale_factor, environment='supervised', data_dir="./data"):
    if environment == 'supervised':
        start_frame_column = 'Starting Frame'
        end_frame_column = 'End Frame Annotation'
    elif environment == 'unsupervised':
        start_frame_column = 'End Frame Annotation'
        end_frame_column = 'Valid until frame'
    else:
        raise ValueError(f'Unknown environment: {environment}')
    settings = pd.read_csv(f'{data_dir}/experiments_settings.csv')
    for id_exp, exp in enumerate(["a", "b", "c", "d", "e"]):
        print(f"Loading experiment {exp}")
        for pos in range(1, 10):
            print(f"Loading position {pos}")
            valid = int(settings[settings.Experiment == f'{exp}{pos}']["Valid"].values[0])
            if valid == 0:
                continue
            start_frame = int(settings[settings.Experiment == f'{exp}{pos}'][start_frame_column].values[0]/reduce_fps_factor)
            end_frame = int(settings[settings.Experiment == f'{exp}{pos}'][end_frame_column].values[0]/reduce_fps_factor)
            if end_frame-start_frame<1:
                continue
            treatment = settings[settings.Experiment == f'{exp}{pos}']['Treatment'].values[0].astype(int)
            fps = settings[settings.Experiment == f'{exp}{pos}']["FPS"].values[0].astype(int)/reduce_fps_factor
            day_hour = settings[settings.Experiment == f'{exp}{pos}']["Hour"].values[0]
            pos_x = settings[settings.Experiment == f'{exp}{pos}']["Position X"].values[0]
            pos_y = settings[settings.Experiment == f'{exp}{pos}']["Position Y"].values[0]

            # load file .mkv
            frames = load_frames(exp, pos, 
                                 reduce_fps_factor=reduce_fps_factor, 
                                 downscale_factor=downscale_factor, 
                                 start_frame=start_frame, 
                                 end_frame=end_frame,
                                 data_dir=data_dir)
            # load annotations
            labels = load_labels(exp, pos, 
                                 reduce_fps_factor=reduce_fps_factor,
                                 start_frame=start_frame,
                                 end_frame=end_frame,
                                 data_dir=data_dir)
            for i in range(end_frame-start_frame):
                yield {
                    "experiment": id_exp,
                    'position': pos,                         
                    "pos_x": pos_x, # covariate                          
                    "pos_y": pos_y, # covariate   
                    "frame": i,
                    "image": frames[i],
                    "treatment": treatment,
                    "outcome": labels[i,:],
                    "exp_minute": ((start_frame+i)/fps)//60, # covariate   
                    "day_hour": day_hour, # covariate   
                }

def map_behaviour_to_label(behaviour):
    if behaviour == ' groom-yellow' or behaviour == ' groom-orange':
        return (1,0)
    if behaviour == ' groom-blue':
        return (0,1)
    if behaviour == ' groom-yellowandblue' or behaviour == ' groom-orangeandblue':
        return (1,1)
    else:
        raise ValueError(f'Unknown behaviour: {behaviour}')

def label_frame(frame_id, behaviors):
    yellow = 0
    blue = 0
    for _, row in behaviors.iterrows():
        if frame_id >= row[' Beginning-frame'] and frame_id < row[' End-frame']:
            yellow_i, blue_i = map_behaviour_to_label(row[' Behavior'])
            yellow += yellow_i
            blue += blue_i
    return (yellow,blue)
        
def load_labels(exp, pos, reduce_fps_factor, start_frame, end_frame, data_dir):
    behaviors_path = os.path.join(data_dir, f"behavior/{exp}{pos}.csv")
    behaviors = pd.read_csv(behaviors_path, skiprows=3, skipfooter=1, engine='python')
    if behaviors.shape[0]==0:
        return torch.zeros(end_frame-start_frame, 2, dtype=torch.float32)
    else:
        labels = []
        for i in range(start_frame, end_frame):
            labels.append(label_frame(i*reduce_fps_factor, behaviors))
        return torch.tensor(labels, dtype=torch.float32)

def load_frames(exp, pos, reduce_fps_factor, downscale_factor, start_frame, end_frame, data_dir="./data"):
    video_name = f'{exp}{pos}.mkv'
    video_path = os.path.join(data_dir, "video", video_name)
    cap = cv2.VideoCapture(video_path)
    #original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Frame rate reduction
        if frame_count % reduce_fps_factor == 0:
            # Downscaling
            if downscale_factor<1:
                resized_frame = cv2.resize(frame, (0, 0), fx=downscale_factor, fy=downscale_factor)
            else: 
                resized_frame = frame
            # Convert to PyTorch tensor (RGB)
            tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1)[[2, 1, 0], :, :]
            frames.append(tensor_frame)
        frame_count += 1

    cap.release()
    return frames[start_frame:end_frame]
