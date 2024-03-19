import numpy as np
import cv2 as cv
import os
from vidaug import augmentors as va
import random as rand
import PIL.Image as Image

print("Initialising video augmentation process.")

raw_folder_path = "data/raw_data"
aug_folder_path = "data/augmented_data"

total_raw_data_files = 0
total_aug_data_files = 0
aug_iterations = 10

for input_folder in os.listdir(raw_folder_path):
    print(f"\tAugmenting video samples in {input_folder}-raw data folder.\n")
    input_folder_path = os.path.join(raw_folder_path, input_folder)
    output_folder_path = os.path.join(aug_folder_path, input_folder)

    # If the folder does not exist, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"\t{input_folder} in augmented data folder created successfully.\n")

    for input_file in os.listdir(input_folder_path):
        print(f"\t\tReading {input_file} video file data.")
        # Import input video file
        input_file_path = os.path.join(input_folder_path, input_file)
        vid_capture = cv.VideoCapture(input_file_path)

        # Get input video file metadata
        frame_width = int(vid_capture.get(3))
        frame_height = int(vid_capture.get(4))
        frame_size = (frame_width,frame_height)
        fps = vid_capture.get(5)

        video = []

        while(vid_capture.isOpened()):
            status, frame = vid_capture.read()
            
            # Compile Video Frames
            if status:
                video.append(Image.fromarray(frame))
            else:
                print(f"\t\t{input_file} frames compiled.")
                break

        vid_capture.release() # Close the video read file object 
        total_raw_data_files += 1

        n = aug_iterations
        print(f"\t\tAugmenting {os.path.splitext(input_file)[0]} video file {n} times.\n")

        for i in range(1, n + 1):
            # Store the list of transformers
            # resize = va.RandomResize(rate= rand.uniform(0.01, 0.10))
            shear = va.RandomShear(x= rand.random() * 0.5, 
                                y= rand.random() * 0.2)
            rotate = va.RandomRotate(degrees= rand.randint(1, 20))
            translate = va.RandomTranslate(x= rand.randint(1, 100), 
                                        y= rand.randint(1, 100))
            downsample = va.Downsample(ratio= 
                                    round(rand.uniform(0.90, 0.99), 
                                            int(np.log10(fps)) + 1))
            upsample = va.Upsample(ratio= 
                                round(rand.uniform(1.01, 1.10), 
                                        int(np.log10(fps)) + 1))
                                                    
            crop = va.RandomCrop(size= 
                                (int(rand.uniform(0.85, 1) * frame_height), 
                                int(rand.uniform(0.85, 1) * frame_width)))

            transformers = [shear, rotate, translate, downsample, upsample, crop]

            # Select augmentor techniques randomly
            augmentors = va.SomeOf(transforms= transformers, 
                                    N= rand.randint(1, 4), 
                                    random_order= True)
            
            aug_vid = augmentors(video)
            print(f"\t\t\tThe {i}-th iteration of {os.path.splitext(input_file)[0]} has been augmented.")

            # Using input video metadata, create MP4 file to store augmented video
            output_file = os.path.splitext(input_file)[0] + "_" + "{:0>3}".format(i) + ".mp4"
            output_file_path = os.path.join(output_folder_path, output_file)

            vid_Write = cv.VideoWriter(output_file_path, 
                                    cv.VideoWriter_fourcc(*'mp4v'), 
                                    fps, 
                                    np.array(aug_vid[0]).shape[1::-1])
            print(f"\t\t\tCreated {output_file} file object.")

            for aug_frame in aug_vid:
                vid_Write.write(np.array(aug_frame))
            
            # Close the video write file object    
            vid_Write.release()
            print(f"\t\t\t{i}-th augmented {os.path.splitext(input_file)[0]} file produced.\n")
            total_aug_data_files += 1

    print(f"\tAll {input_file} data files augmented successfully.\n")

print(f"Augmentation algorithm has run to completion. Total of {total_raw_data_files} augmented for {aug_iterations} times, for a total of {total_aug_data_files} augmented files produced.")