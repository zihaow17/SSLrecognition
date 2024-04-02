import sys
sys.path.append("d:\\SMU\\ml&applns")

from Preprocessing.Video_Preprocessing import VideoPreprocessing
from Preprocessing.DFTransformations import *
from Preprocessing.DataExtration import *
from Preprocessing.Average_parquet import Averager
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import pandas as pd
import numpy as np
import warnings
import pickle

class Pipeline():

    def __init__(self, path_to_video) -> None:
        self.frames_target = 65
        self.duplicate = True
        self.remove_points = True

        with open('../Preprocessing/standard_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        data = VideoPreprocessing().process_video(path_to_video)
        self.data = Averager(data).average_pf()
        pass

    def process_input(self) -> np.ndarray:
        self.data_restructuring()
        self.data_statistics()
        return self.data

    def data_restructuring(self) -> None:
        video_data = self.data

        # Remove all rows with the type of face
        video_data = video_data.drop(video_data.loc[video_data.type=="face"].index)

        # Replace all NaN values with 0
        video_data = video_data.fillna(0)

        video_data = transform_data(video_data, self.frames_target, self.duplicate, self.remove_points)
        self.data = video_data

    def data_statistics(self) -> None:
        video_data = self.data

        video_data = video_data.sort_values('type')

        if self.remove_points:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_data = calc_stats_types(video_data)
        else:
            video_data = calculate_stats(video_data)

        if not self.duplicate:
            frame_num = video_data.shape[0]
            if frame_num != self.frames_target:
                pad = np.zeros(shape=((self.frames_target - frame_num), video_data.shape[1]))
                video_data = np.concatenate((video_data, pad), axis=0)

        video_data = np.array(video_data)
        video_data = np.reshape(video_data, (1, 65, 120))
        
        # Scaling the data
        video_data_std = video_data.reshape(video_data.shape[0], -1)
        video_data_std = self.scaler.fit_transform(video_data_std)
        video_data = video_data_std.reshape(video_data.shape)

        self.data = video_data