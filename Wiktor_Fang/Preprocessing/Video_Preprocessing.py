import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd
from Preprocessing.mp_support import mediapipe_detection, draw_landmarks, draw_styled_landmarks, extract_keypoints, extract_coordinates, prob_viz

class VideoPreprocessing():

    def __init__(self) -> None:
        self.savepath = "../Pipeline/video"
        self.clean_folder()
        pass

    def clean_folder(self) -> None:
        # Get a list of all files in the folder
        files = os.listdir(self.savepath)

        # Iterate over the files and remove each one
        for file in files:
            file_path = os.path.join(self.savepath, file)
            os.remove(file_path)

    def process_video(self, video_path) -> pd.DataFrame:
        self.extract_frames_data(video_path)
        return self.convert_to_df()

    def extract_frames_data(self, path) -> None:
        extract_coordinates(path, self.savepath)

    def convert_to_df(self) -> pd.DataFrame:

        # create empty data frame
        df = pd.DataFrame(columns=['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])
        
        frame_counter = 1 
        
        # for video
        for frame in os.listdir(self.savepath):
            
            frame_path = os.path.join(self.savepath, frame)
            
            with open(frame_path, 'rb') as file:
                npy_data = np.load(file)
            
                # scan through npy file and add data to df in batches of 3
                local_count = 1
                for j in range(0, 99, 3):
                    new_row = pd.Series({'frame': frame_counter, 'row_id': f"{frame_counter}-pose-{local_count}", 'type': 'pose', 'landmark_index': local_count, 'x': npy_data[j], 'y': npy_data[j+1], 'z': npy_data[j+2]})
                    
                    df = pd.concat([df, new_row.to_frame().T], ignore_index = True, axis = 0)
                    
                    local_count += 1
                
                local_count = 1
                for j in range(99, 162, 3):
                    new_row = pd.Series({'frame': frame_counter, 'row_id': f"{frame_counter}-left_hand-{local_count}", 'type': 'left_hand', 'landmark_index': local_count, 'x': npy_data[j], 'y': npy_data[j+1], 'z': npy_data[j+2]})
                    
                    df = pd.concat([df, new_row.to_frame().T], ignore_index = True, axis = 0)
                    
                    local_count += 1
                
                local_count = 1
                for j in range(162, 225, 3):
                    new_row = pd.Series({'frame': frame_counter, 'row_id': f"{frame_counter}-right_hand-{local_count}", 'type': 'right_hand', 'landmark_index': local_count, 'x': npy_data[j], 'y': npy_data[j+1], 'z': npy_data[j+2]})
                    
                    df = pd.concat([df, new_row.to_frame().T], ignore_index = True, axis = 0)
                    
                    local_count += 1
                
            frame_counter += 1
        
        return df