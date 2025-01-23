"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import os
import numpy as np
import json
from tqdm import tqdm


# Let's take this one for this example
# repo_id = "jainamit/koch"
repo_id = "jainamit/koch_realcube3"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id)

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
pprint(ds_meta.features)

# You can also get a short summary by simply printing the object:
print(ds_meta)

# # You can then load the actual dataset from the hub.
# # Either load any subset of episodes:
# dataset = LeRobotDataset(repo_id, episodes=[0, 1])

# # And see how many frames you have:
# print(f"Selected episodes: {dataset.episodes}")
# print(f"Number of episodes selected: {dataset.num_episodes}")
# print(f"Number of frames selected: {dataset.num_frames}")

# load the entire dataset:
dataset = LeRobotDataset(repo_id)  # Load all episodes, not just [0,1]
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# The previous metadata class is contained in the 'meta' attribute of the dataset:
print(dataset.meta)

# LeRobotDataset actually wraps an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets for more information).
print(dataset.hf_dataset)

# LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
# with the latter, like iterating through the dataset.
# The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
# episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
# frame indices associated to the first episode:
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# Then we grab all the image frames from the first camera:
camera_key = dataset.meta.camera_keys[0]
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# The objects returned by the dataset are all torch.Tensors
print(type(frames[0]))
print(frames[0].shape)

# Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
# We can compare this shape with the information available for that feature
pprint(dataset.features[camera_key])
# In particular:
print(dataset.features[camera_key]["shape"])
# The shape is in (h, w, c) which is a more universal format.




def prepare_episode_data(frames, states, actions, language_instruction="Pick up the cube", embedding_dim=512):
    """Prepare episode data from frames, states, and actions."""
    episodes = []
    num_frames = len(frames)
    
    for frame_idx in range(num_frames):
        # Convert torch tensor to numpy 
        frame = frames[frame_idx].numpy()   # Its normalized : frame.max() = 1
            
        # Create episode entry with correct observation structure
        episode = {
            'observation': {
                'images': {   
                    'front_camera': np.transpose(frame, (1, 2, 0)).astype(np.float32),    # Convert to HWC format
                },
                'state': states[frame_idx].numpy().astype(np.float32),
            },
            'action': actions[frame_idx].numpy().astype(np.float32),
            'language_instruction': language_instruction,
            'language_embedding': np.random.rand(embedding_dim).astype(np.float32),
            'discount': 1.0,
            'reward': float(frame_idx == num_frames - 1),
            'is_first': frame_idx == 0,
            'is_last': frame_idx == num_frames - 1,
            'is_terminal': frame_idx == num_frames - 1,
        }
        episodes.append(episode)
    
    return episodes
 

def save_complete_dataset(dataset, base_output_dir, ds_meta):
    """Save complete dataset from HuggingFace repo to local storage."""
    
    # Create directory structure
    os.makedirs(base_output_dir, exist_ok=True)
    episodes_dir = os.path.join(base_output_dir, 'train')  # can also add validation and test splits if needed
    os.makedirs(episodes_dir, exist_ok=True)
    
    # Save metadata using ds_meta
    metadata = {
        'num_episodes': ds_meta.total_episodes,
        'camera_keys': ds_meta.camera_keys,
        'feature_info': ds_meta.features,
        'fps': ds_meta.fps,
        'robot_type': ds_meta.robot_type,
        'tasks': ds_meta.tasks
    }
    with open(os.path.join(base_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save all episodes
    for episode_idx in tqdm(range(ds_meta.total_episodes), desc="Saving episodes"):
        # Get episode frame indices
        from_idx = dataset.episode_data_index["from"][episode_idx].item()
        to_idx = dataset.episode_data_index["to"][episode_idx].item()
        
        # Extract data
        camera_key = dataset.meta.camera_keys[0]
        frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]
        states = [dataset[idx]['observation.state'] for idx in range(from_idx, to_idx)]
        actions = [dataset[idx]['action'] for idx in range(from_idx, to_idx)]
        
        # Prepare episode data
        episodes = prepare_episode_data(frames, states, actions)
        
        # Save episode file
        episode_filename = os.path.join(episodes_dir, f'episode_{episode_idx}.npy')
        np.save(episode_filename, episodes)
        print(f"Saved episode {episode_idx} to {episode_filename}")

    print(f"Successfully saved {ds_meta.total_episodes} episodes to {episodes_dir}")

# Usage
output_directory = "/mnt/data/amit/Datasets/raw_datasets/lerobot"
save_complete_dataset(dataset, output_directory, ds_meta)

