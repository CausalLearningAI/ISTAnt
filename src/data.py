import torch
import pandas as pd
import cv2
import os
from datasets import Dataset

def get_data_cl(environment="supervised", data_dir="./data", task="all"):
    data = load_data(environment=environment, data_dir=data_dir, generate=False)
    covariates = ['pos_x', 'pos_y', 'exp_minute', 'day_hour']
    X = torch.stack([torch.tensor(data[covariate]) for covariate in covariates], dim=1)
    t = data["treatment"]
    if task=="all":
        y = data["outcome"]
    elif task.lower()=="yellow":
        y = data["outcome"][:,0]
    elif task.lower()=="blue":
        y = data["outcome"][:,1]
    elif task.lower()=="sum":
        y = data["outcome"].sum(axis=1)
    else:
        raise ValueError(f"Task {task} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    return X, y, t

def get_examples(environment="supervised", data_dir="./data", n=36, task="all", encoder_name="dino"):
    data = load_data(environment=environment, data_dir=data_dir, generate=False)
    idxs = torch.randint(0, len(data), (n,))
    image = data[idxs]["image"]
    if environment=="supervised":
        if task=="all":
            y = data[idxs]["outcome"]
        elif task.lower()=="yellow":
            y = data[idxs]["outcome"][:,0]
        elif task.lower()=="blue":
            y = data[idxs]["outcome"][:,1]
        elif task.lower()=="sum":
            y = data[idxs]["outcome"].sum()
        else:
            raise ValueError(f"Task {task} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    else:
        y = None
    del data
    data_emb_env_dir = os.path.join(data_dir, encoder_name, environment)
    if not os.path.exists(data_emb_env_dir):
        raise Exception(f"`Embedding` {encoder_name} has not been extracted yet for `environment` {environment}, or just doesn't exist. Please select `encoder_name` and `environment` with valid embeddings extracted.")
    embeddings = Dataset.load_from_disk(data_emb_env_dir)
    embedding = embeddings[idxs][encoder_name]
    return image, y, embedding

    
def get_data_sl(environment="supervised", encoder_name="vit", data_dir="./data/", task="all", split_criteria="experiment"):
    data = load_data(environment=environment, data_dir=data_dir, generate=False)
    data_emb_env_dir = os.path.join(data_dir, encoder_name, environment)
    embedding = Dataset.load_from_disk(data_emb_env_dir)
    X = embedding[encoder_name]
    if task=="all":
        y = data["outcome"]
    elif task.lower()=="yellow":
        y = data["outcome"][:,0]
    elif task.lower()=="blue":
        y = data["outcome"][:,1]
    elif task.lower()=="sum":
        y = data["outcome"].sum(axis=1)
    else:
        raise ValueError(f"Task {task} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    if split_criteria=="experiment":
        split = (data["experiment"] == 0)
    elif split_criteria=="position":
        split = (data["position"] == 0)
    else:
        raise ValueError(f"Split criteria {split_criteria} doesn't exist. Please select a valid splitting criteria: 'experiment', 'position'.")
    return X, y, split

def load_data(environment='supervised', data_dir="./data", generate=False, reduce_fps_factor=10, downscale_factor=0.4):
    data_env_dir = os.path.join(data_dir, environment)
    if generate:
        dataset = Dataset.from_generator(generator, gen_kwargs={"reduce_fps_factor": reduce_fps_factor, "downscale_factor": downscale_factor, "environment":environment, "data_dir":data_dir})
        dataset.save_to_disk(data_env_dir)
        dataset.set_format(type="torch", columns=["image", "treatment", "outcome", 'pos_x', 'pos_y', 'exp_minute', 'day_hour', 'frame', "experiment", "position"], output_all_columns=True)
    else:
        if not os.path.exists(data_env_dir):
            raise Exception("The dataset is not saved, please set generate=True to generate the dataset, or correct the path_dir.")
        dataset = Dataset.load_from_disk(data_env_dir)
        dataset.set_format(type="torch", columns=["image", "treatment", "outcome", 'pos_x', 'pos_y', 'exp_minute', 'day_hour', 'frame', "experiment", "position"], output_all_columns=True)
    return dataset

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
            if (exp == "c" and pos == 9):
                continue
            start_frame = int(settings[settings.Experiment == f'{exp}{pos}'][start_frame_column].values[0]/reduce_fps_factor)
            end_frame = int(settings[settings.Experiment == f'{exp}{pos}'][end_frame_column].values[0]/reduce_fps_factor)
            treatment = settings[settings.Experiment == f'{exp}{pos}']['Treatment'].values[0].astype(int)
            fps = settings[settings.Experiment == f'{exp}{pos}']["Frame Rate (FPS)"].values[0].astype(int)/reduce_fps_factor
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
                                 end_frame=end_frame)
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
    if behaviour == ' groom-yellow':
        return (1,0)
    if behaviour == ' groom-blue':
        return (0,1)
    if behaviour == ' groom-yellowandblue':
        return (1,1)
    else:
        raise ValueError(f'Unknown behaviour: {behaviour}')

def label_frame(frame_id, behaviors):
    for _, row in behaviors.iterrows():
        if frame_id >= row[' Beginning-frame'] and frame_id < row[' End-frame']:
            return map_behaviour_to_label(row[' Behavior'])
    return (0,0)
        
def load_labels(exp, pos, reduce_fps_factor, start_frame, end_frame):
    behaviors = pd.read_csv(f'./data/behavior/{exp}{pos}.csv', skiprows=3, skipfooter=1, engine='python')
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
