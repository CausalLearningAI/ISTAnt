import torch
import pandas as pd
import cv2


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
            resized_frame = cv2.resize(frame, (0, 0), fx=downscale_factor, fy=downscale_factor)
            # Convert to PyTorch tensor
            tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1)  
            frames.append(tensor_frame)
        frame_count += 1

    cap.release()
    return frames[start_frame:end_frame]
