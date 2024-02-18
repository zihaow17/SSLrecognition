import pandas as pd
import numpy as np

class Averager():
    # Initialize and fetch class data
    def __init__(self, pf) -> None:
        self.pf = pf.fillna(-1)
 
        # Remove random instances of type/s which in frames, if not present in other frames
        self.pf = self.del_artifacts(self.pf)
        # Generate dictionary of missing values
        self.nas_dict = self.gen_dict_of_nas_vals(self.pf)
        pass

    # Remove random instances of type/s which in frames, if not present in other frames
    def del_artifacts(self, pf):
        # Fetch the types of objects in dataset
        points_types = pf['type'].unique()
        # Calculate amount of frames
        cnt_frames = max(pf['frame']) - min(pf['frame'])

        # Loop through types to check for artifacts
        for point_type in points_types:
            # Count frames containing position data for given type
            frames_for_type = len(pf['frame'].loc[(pf.type==point_type) & (pf.x > 0)].unique())

            # Check if data exists for a minimum of 10% of frames
            if frames_for_type < int(cnt_frames/10):
                # If less than or equal to 10%, replace with numpy NaN
                pf.loc[pf['type']==point_type, 'x'] = np.nan
                pf.loc[pf['type']==point_type, 'y'] = np.nan
                pf.loc[pf['type']==point_type, 'z'] = np.nan

        # Return cleaned dataset
        return pf

    # Function for generating a dictionary of values to be recalculated
    # The dictionary is of structure-
    #   {frame_number:
    #       {type_of_point: 
    #           [landmark_indices_of_points]
    #       }
    #   }
    def gen_dict_of_nas_vals(self, pf):
        # Create an empty dictionary
        nas_dict = {}
        # Fetch list of indices for points with missing data
        nas_indices = list(pf.loc[pf['x'] < 0].index)

        # Iterate through index values stored in nas_indices
        for idx in nas_indices:
            # Fetch the row data located at index idx
            row = pf.iloc[idx]

            # Extract "frame", "type" and "landmark_index" from row
            frame = row.frame
            pt_type = row.type
            ldmrk_index = row.landmark_index

            # Check if nas_dict contains the "frame" number in it's keys
            if frame in nas_dict.keys():
                # If true:
                # Check if inner dictionare contains the point "type" in it's keys
                if pt_type in nas_dict[frame].keys():
                    # Append index to list if true
                    nas_dict[frame][pt_type].append(ldmrk_index)
                else:
                    # If false, create a new list with index and assign to inner dictionary at key: pt_type
                    nas_dict[frame][pt_type] = [ldmrk_index]
            else:
                # If false, create a new dictionary containing pt_type as key and list with index inside, and assign at key: frame
                nas_dict[frame] = {pt_type: [ldmrk_index]}

        # Return dictionary
        return nas_dict
    
    # Calculate average positional data for missing points
    def average_pf(self):
        # Assign class variables to local variables
        pf = self.pf
        nas_dict = self.nas_dict

        # Generate placeholder dictionary of closest next valid positions available
        next_positions_dict = self.gen_dict_of_next_pos(pf)

        # Fetch the minimum value for frame
        min_frame = min(pf.frame)

        # Iterate through the frames in nas_dict
        for frame in nas_dict.keys():
            # Iterate through the types in the inner dictionary in nas_dict
            for pt_type in nas_dict[frame].keys():
                # Extract list of landmark positions from dictionary
                lndmrks_pos = nas_dict[frame][pt_type]

                # Check if the next available positional values are from a frame earlier than current
                if next_positions_dict[pt_type]['frame'] <= frame:
                    # If true, update dictionary with next available positional values
                    self.update_next_pos(next_positions_dict, pt_type, frame, pf)

                # Extract values of x,y and z coordinates from next_positions_dict
                next_x = next_positions_dict[pt_type]["x"]
                next_y = next_positions_dict[pt_type]["y"]
                next_z = next_positions_dict[pt_type]["z"]
                
                # Check if current frame is the first frame
                if frame == min_frame:
                    # If true, set new positional values to next values
                    new_x = next_x
                    new_y = next_y
                    new_z = next_z
                else:
                    # If false, fetch positional values for previous frame
                    # Create placeholder value for prev_x
                    prev_x = pd.Series([]).values
                    # Set step to 1
                    step = 1

                    # Loop until prev_x is not empty
                    # In very rare cases, certain parquets are missing frames in between, therefore this loop is required to prevent any issues
                    while prev_x.size == 0:
                        # Fetch positional values for previous frame
                        prev_x, prev_y, prev_z = self.fetch_coords(frame-step, pt_type, pf)
                        # Increment step by 1
                        step += 1

                    # Calculate difference between current frame and final frame
                    frame_diff = (next_positions_dict[pt_type]["frame"] - frame) + 1

                    # Calculate new positional values
                    # Explanation: frame_diff is used here to offset the value by the amount of empty frames next
                    #       Example: 20 __ __ __ 40
                    #           20 is at pos 0; 40 is at pos 4
                    #           We want the value for pos 1, therefore we calculate difference between 40 and 20 = 20
                    #           Using above formula: frame_diff = (4 - 1) + 1 = 4
                    #           The new value will be: 20 + (20/4 = 5) = 25 :   20 25 __ __ 40
                    #           This creates an even spacing between values, hence "smooth" transition from one frame to another
                    new_x = prev_x + (next_x - prev_x)/frame_diff
                    new_y = prev_y + (next_y - prev_y)/frame_diff
                    new_z = prev_z + (next_z - prev_z)/frame_diff

                # Update positional values at frame, type and landmark_index provided
                pf.loc[(pf.frame == frame) & (pf.type == pt_type) & (pf.landmark_index.isin(nas_dict[frame][pt_type])), 'x'] = [new_x[i] for i in lndmrks_pos]
                pf.loc[(pf.frame == frame) & (pf.type == pt_type) & (pf.landmark_index.isin(nas_dict[frame][pt_type])), 'y'] = [new_y[i] for i in lndmrks_pos]
                pf.loc[(pf.frame == frame) & (pf.type == pt_type) & (pf.landmark_index.isin(nas_dict[frame][pt_type])), 'z'] = [new_z[i] for i in lndmrks_pos]

        # Return Averaged pf
        return pf

    # Generate a placeholder dictionary for next positional values
    def gen_dict_of_next_pos(self, pf):
        # Fetch all types in dataset
        points_types = pf.type.unique()
        # Create an empty dictionary
        next_positions_dict = {}

        # Iterate through list of types and update dictionary
        for point_type in points_types:
            # Update dictionary at key value: "type" with placeholder dictionary
            next_positions_dict[point_type] = {"x": [], "y": [], "z": [], "frame": 0}

        # Return dictionary
        return next_positions_dict

    # Update next positional values for given type
    def update_next_pos(self, next_positions_dict, pt_type, frame, pf):
        # Fetch first and last frame numbers
        max_frame = max(pf.frame)
        min_frame = min(pf.frame)
        # Initialize a counter "step" to 1
        step = 1

        # Loop until break
        while 1:
            # Check if frame + step is equal to or smaller than the last frame
            if frame+step <= max_frame:
                # If true, fetch positional values for type at frame + step
                next_x, next_y, next_z = self.fetch_coords(frame+step, pt_type, pf)
            else:
                # If false, this means that no frames in front contain any positional data or current frame is the last frame
                # Check if frame is the first frame
                if frame == min_frame:
                    # If true, set next positional values to a list of -1
                    next_x = next_y = next_z = [-1 for i in range(max(pf['landmark_index'].loc[pf.type==pt_type])+1)]
                else:
                    # If false, set next positional values to values of previous frame
                    # Create placeholder value for next_x
                    next_x = pd.Series([]).values
                    # Initialize a counter "i_step" to 1
                    i_step = 1
                    
                    # Loop until next_x is not empty
                    # In very rare cases, certain parquets are missing frames in between, therefore this loop is required to prevent any issues
                    while next_x.size == 0:
                        # Fetch positional values for previous frame
                        next_x, next_y, next_z = self.fetch_coords(frame-i_step, pt_type, pf)
                        # Increment i_step by 1
                        i_step += 1
                
                # Update next positional values within the dictionary
                next_positions_dict[pt_type]["x"] = next_x
                next_positions_dict[pt_type]["y"] = next_y
                next_positions_dict[pt_type]["z"] = next_z
                # Update frame value in dictionary to last frame number, as no frames in front store any data of interest
                # Prevents unnecesary looping
                next_positions_dict[pt_type]["frame"] = max_frame
                # Break out of loop
                break

            # Check if fetched values in next_x are all values below 0 and the list next_x is not empty
            if (next_x < 0).all() or next_x.size == 0:
                # If true, increment step by 1
                step += 1
            else:
                # If false, update next positional within the dictionary
                next_positions_dict[pt_type]["x"] = next_x
                next_positions_dict[pt_type]["y"] = next_y
                next_positions_dict[pt_type]["z"] = next_z
                # Update frame value in dictionary to frame number (frame + step) of extracted values
                next_positions_dict[pt_type]["frame"] = frame+step
                # Break out of loop
                break

    # Fetch positional data of type for provided frame
    def fetch_coords(self, frame, point_type, pf):
        # Fetch data of object in frame + direction
        step_pf = pf.loc[(pf['frame']==frame) & (pf['type']==point_type)]
        # Extract positional data
        x, y, z = step_pf['x'], step_pf['y'], step_pf['z']
        # Return positional data as numpy.ndarray
        return x.values,y.values,z.values