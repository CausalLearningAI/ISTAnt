import torch
import pandas as pd
import cv2
import os
from datasets import Dataset

def get_data_cl(environment="supervised", data_dir="./data/", outcome="all"):
    data = load_data(environment=environment, path_dir=data_dir, generate=False)
    X = None
    t = data["treatment"]
    if outcome=="all":
        y = data["outcome"]
    elif outcome.lower()=="yellow":
        y = data["outcome"][:,0]
    elif outcome.lower()=="blue":
        y = data["outcome"][:,1]
    elif outcome.lower()=="sum":
        y = data["outcome"].sum(axis=1)
    else:
        raise ValueError(f"Outcome {outcome} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    return X, y, t

def get_examples(environment="supervised", data_dir="./data/", idxs=[600], outcome="all", model_name="dino"):
    data = load_data(environment=environment, path_dir=data_dir, generate=False)
    image = data[idxs]["image"]
    if outcome=="all":
        y = data[idxs]["outcome"]
    elif outcome.lower()=="yellow":
        y = data[idxs]["outcome"][:,0]
    elif outcome.lower()=="blue":
        y = data[idxs]["outcome"][:,1]
    elif outcome.lower()=="sum":
        y = data[idxs]["outcome"].sum()
    else:
        raise ValueError(f"Outcome {outcome} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    del data
    embeddings = Dataset.load_from_disk(f'{data_dir}{model_name}/{environment}')
    embedding = embeddings[idxs][model_name]
    return image, y, embedding

    
def get_data_sl(environment="supervised", model_name="vit", data_dir="./data/", outcome="all"):
    data = load_data(environment=environment, path_dir=data_dir, generate=False)
    embedding = Dataset.load_from_disk(f'{data_dir}{model_name}/{environment}')
    X = embedding[model_name]
    if outcome=="all":
        y = data["outcome"]
    elif outcome.lower()=="yellow":
        y = data["outcome"][:,0]
    elif outcome.lower()=="blue":
        y = data["outcome"][:,1]
    elif outcome.lower()=="sum":
        y = data["outcome"].sum(axis=1)
    else:
        raise ValueError(f"Outcome {outcome} not defined. Please select between: 'all', 'yellow', 'blue', 'sum'.")
    return X, y 

def load_data(environment='supervised', path_dir="./data/", generate=False, reduce_fps_factor=10, downscale_factor=0.4):
    if generate:
        dataset = Dataset.from_generator(generator, gen_kwargs={"reduce_fps_factor": reduce_fps_factor, "downscale_factor": downscale_factor, "environment":environment})
        dataset.save_to_disk(path_dir+environment)
        dataset.set_format(type="torch", columns=["image", "treatment", "outcome"], output_all_columns=True)
    else:
        # check if the dataset is already saved
        if not os.path.exists(path_dir+environment):
            raise Exception("The dataset is not saved, please set generate=True to generate the dataset, or correct the path_dir.")
        dataset = Dataset.load_from_disk(path_dir+environment)
        dataset.set_format(type="torch", columns=["image", "treatment", "outcome"], output_all_columns=True)
    return dataset

def generator(reduce_fps_factor, downscale_factor, environment='supervised'):
    if environment == 'supervised':
        start_frame_column = 'Starting Frame'
        end_frame_column = 'End Frame Annotation'
    elif environment == 'unsupervised':
        start_frame_column = 'End Frame Annotation'
        end_frame_column = 'Valid until frame'
    else:
        raise ValueError(f'Unknown environment: {environment}')
    settings = pd.read_csv(f'./data/experiments_settings.csv')
    for exp in ["a", "b", "c", "d", "e"]:
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
            pos_y = settings[settings.Experiment == f'{exp}{pos}']["Position X"].values[0]

            # load file .mkv
            frames = load_frames(exp, pos, 
                                 reduce_fps_factor=reduce_fps_factor, 
                                 downscale_factor=downscale_factor, 
                                 start_frame=start_frame, 
                                 end_frame=end_frame)
            # load annotations
            labels = load_labels(exp, pos, 
                                 reduce_fps_factor=reduce_fps_factor,
                                 start_frame=start_frame,
                                 end_frame=end_frame)
            for i in range(end_frame-start_frame):
                yield {
                    "experiment": exp,
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

def load_frames(exp, pos, reduce_fps_factor, downscale_factor, start_frame, end_frame):
    video_path = f'./data/video/{exp}{pos}.mkv'
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
            # Convert to PyTorch tensor
            tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1)[[2, 1, 0], :, :]
            frames.append(tensor_frame)
        frame_count += 1

    cap.release()
    return frames[start_frame:end_frame]
