import pandas as pd
import numpy as np

def transform_data(pf, frame_amt_goal, duplicate=False, remove_points=False):
    if remove_points:
        pf = __remove_points(pf)
        
    pf = __reset_frame_nums(pf)
    frame_nums = pf.frame.unique()
    frame_diff = abs(frame_amt_goal - len(frame_nums))
    operation = frame_amt_goal > len(frame_nums)

    values_to_operate = np.linspace(0, len(frame_nums) - 1, frame_diff, dtype=int)

    if operation:
        if duplicate:
            pf = duplicate_vals(pf, values_to_operate)
    else:
        pf = remove_vals(pf, values_to_operate)
        
    return pf

def __remove_points(pf):
    pose_indices = [0,2,5,7,8,9,10,11,12,13,14,23,24]
    hand_indices = [0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20]
    row_ids = []

    for frame in pf.frame.unique():
        for i in range(len(hand_indices)):
            if i < len(pose_indices):
                row_ids.append(f'{frame}-pose-{pose_indices[i]}')
            row_ids.append(f'{frame}-right_hand-{hand_indices[i]}')
            row_ids.append(f'{frame}-left_hand-{hand_indices[i]}')

    pf = pf.loc[(pf.row_id.isin(row_ids))].copy()
    pf = __pose_to_face_points(pf)
    
    return pf

def __pose_to_face_points(pf):
    face_landmarks = [0,2,5,7,8,9,10]

    face_points = pf.loc[(pf.type=="pose")&(pf.landmark_index.isin(face_landmarks))].copy()
    face_points.type = "face"
    face_points.row_id = face_points.frame.astype(str) + '-' + face_points.type + '-' + face_points.landmark_index.astype(str)

    pf.loc[face_points.index] = face_points

    return pf

def __reset_frame_nums(pf):
    # Define the shape of the array
    shape_of_frame_vals = (len(pf.frame.unique()), sum(pf.frame==min(pf.frame)))

    # Create the array using broadcasting
    result_array = np.arange(shape_of_frame_vals[0])[:, np.newaxis]

    # Repeat the values along the second dimension
    result_array = np.tile(result_array, shape_of_frame_vals[1])

    # Concatenated array into list of new frame numbers
    new_frame_values = np.concatenate(result_array)

    pf.frame = new_frame_values

    return pf

def duplicate_vals(pf, values):
    offset = 0
    for val in np.unique(values):
        cnt = sum(values==val)
        if offset < cnt:
            offset = cnt
    offset += 1
    
    pf.frame *= offset
    values *= offset

    values = list(values)

    duplicate_frames = pd.DataFrame()

    i = 1
    while len(values) != 0:
        frame_nums = np.unique(values)

        dup_frames = pf.loc[pf.frame.isin(frame_nums)].copy()
        dup_frames.frame += i
        duplicate_frames = pd.concat([duplicate_frames, dup_frames])

        i += 1

        for num in frame_nums:
            values.remove(num)

    pf = pd.concat([pf, duplicate_frames], ignore_index=True).reset_index(drop=True).sort_values('frame')
    return pf

def pad_0(pf, amount):
    frame = pf.loc[pf.frame==min(pf.frame)].copy()
    frame *= 0
    max_frame = max(pf.frame)
    
    for i in range(amount):
        frame.frame = max_frame+i+1
        pf = pd.concat([pf, frame], ignore_index=True)

    pf = pf.reset_index(drop=True).sort_values('frame')
    return pf


def remove_vals(pf, values: np.array):
    pf = pf.drop(pf.loc[pf.frame.isin(values)].index)
    return pf