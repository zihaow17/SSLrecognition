import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def calculate_stats(df):
        grouped_data = df.groupby('frame')[['x','y','z']]
        grouped_data = np.array([group.values for _, group in grouped_data])
        data = np.transpose(grouped_data, (0, 2, 1))

        video_data = calculate_stats_np(data)

        return video_data

def calculate_stats_np(data):
    means = np.mean(data, axis=2)
    stds = np.std(data, axis=2)
    cvs = stds/means
    vars = np.var(data, axis=2)
    medians = np.median(data, axis=2)
    mins = np.min(data, axis=2)
    maxs = np.max(data, axis=2)
    skews = skew(data, axis=2)
    kurtosiss = kurtosis(data, axis=2)
    entropies = entropy(abs(data), axis=2)

    video_data = np.stack([means, stds, cvs, vars, medians, mins, maxs, skews, kurtosiss, entropies], axis=-1).reshape((data.shape[0],-1))
    return video_data

def calc_stats_types(pf):
    grouped_data = pf.groupby('frame')[['x','y','z']]
    grouped_data = np.array([group.values for _, group in grouped_data])
    arr = np.transpose(grouped_data, (0, 2, 1))

    face_data = arr[:, :, :7]
    left_hand_data = arr[:, :, 7:24]
    pose_data = arr[:, :, 24:30]
    right_hand_data = arr[:, :, 30:]

    data = [left_hand_data, pose_data, right_hand_data]

    video_data = calculate_stats_np(face_data)

    for a in data:
        video_data = np.concatenate((video_data, calculate_stats_np(a)), axis=1)

    video_data = np.nan_to_num(video_data, 0)

    return video_data

def create_data_grids(df, grid_size=32):
    frames_data = []

    for frame in df.frame.unique():
        frame_data = []

        for data_type in df.type.unique():
            data = df.loc[(df.frame==frame)&(df.type==data_type)]

            x = data['x'].values
            y = data['y'].values

            x[x < 0] = 0
            y[y < 0] = 0

            grid = np.zeros((grid_size,grid_size))

            x = np.round(normalize_list(x, grid_size)).astype(int)
            y = np.round(normalize_list(y, grid_size)).astype(int)

            grid[x, y] += data['z']

            frame_data.append(grid)

        frames_data.append(frame_data)

    frames_data = np.transpose(np.array(frames_data), (0, 2, 3, 1))
    return frames_data

def normalize_list(values, limit):
    min_val = min(values)
    max_val = max(values)
    
    # Handle the case where max and min are equal
    if max_val == min_val:
        normalized_values = [0 if x == min_val else 1 for x in values]
    else:
        normalized_values = [((x - min_val) / (max_val - min_val))*(limit-1) for x in values]
    
    return normalized_values