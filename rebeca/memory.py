import os
import glob
import json
import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
import torch
import faiss

from openai_vpt.lib.policy import MinecraftPolicy
from openai_vpt.lib.tree_util import tree_map

from model import VPTEncoder
from action_utils import ActionProcessor


class SituationLoader:
    """Data loader for loading expert demonstrations and creating situation embeddings"""

    def __init__(self, vpt_model: VPTEncoder, data_dir="data/MakeWaterfall/"):
        self.vpt = vpt_model
        self.action_processor = ActionProcessor()
        self.load_expert_data(data_dir)

    def load_expert_data(self, data_dir):
        """Load expert demonstrations from data_dir"""

        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        unique_ids.sort()

        self.demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(data_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(data_dir, unique_id + ".jsonl"))
            self.demonstration_tuples.append((unique_id, video_path, json_path))

    def load_demonstrations(self, num_demos=None):
        """Load expert demonstrations from demonstration tuples"""

        if num_demos is not None:
            _demonstration_tuples = self.demonstration_tuples[:num_demos]
        else:
            _demonstration_tuples = self.demonstration_tuples

        demonstrations = []
        for unique_id, video_path, json_path in tqdm(
            _demonstration_tuples, desc="Loading expert demonstrations"
        ):
            video = self._load_video(video_path)
            jsonl = self._load_jsonl(json_path)
            demonstrations.append(
                {"demo_id": unique_id, "video": video, "jsonl": jsonl}
            )
        return demonstrations

    def encode_demonstrations(self, demonstrations):
        encoded_demos = []
        for demo in tqdm(demonstrations, desc="Encoding expert demonstrations"):
            encoded_demo = self.vpt.encode_trajectory(demo["video"])
            encoded_demos.append(
                {"demo_id": demo["demo_id"], "encoded_demo": encoded_demo, "actions": demo["jsonl"]}
            )
        return encoded_demos

    def load_encode_save_demos(self, num_demos=None, save_dir="data/MakeWaterfallEncoded/"):
        '''Load, encode and save expert demonstrations to disk'''
        
        # Select the number of demonstrations to load
        if num_demos is not None:
            _demonstration_tuples = self.demonstration_tuples[:num_demos]
        else:
            _demonstration_tuples = self.demonstration_tuples

        # Create a directory to save the encoded demonstrations
        os.makedirs(save_dir, exist_ok=True)
        
        for unique_id, video_path, json_path in tqdm(
            _demonstration_tuples, desc="Loading expert demonstrations"
        ):
            video = self._load_video(video_path)
            jsonl = self._load_jsonl(json_path)
            
            # Encode the demonstration
            encoded_demo = self.vpt.encode_trajectory(video, tolist=True)

            encoded_demo_json = {"demo_id": unique_id, "encoded_demo": encoded_demo, "actions": jsonl}

            # Save the encoded demonstration to disk
            with open(os.path.join(save_dir, unique_id + ".pkl"), "wb") as f:
                pickle.dump(encoded_demo_json, f)

    
    def load_encoded_demos_to_situations(self, save_dir="data/MakeWaterfallEncoded/", window_size=128, stride=2):
        '''Load encoded demonstrations from disk and create situations'''
        
        situations = []
        for pkl_path in tqdm(glob.glob(os.path.join(save_dir, "*.pkl")), desc="Loading encoded demonstrations"):
            with open(pkl_path, "rb") as f:
                demo = pickle.load(f)
                for i in range(
                    window_size, len(demo["encoded_demo"]) - window_size, stride
                ):
                    situations.append(
                        {
                            "demo_id": demo["demo_id"],
                            "sit_frame_idx": i, # Frame index of the situation in the video
                            "situation": demo["encoded_demo"][i],
                            "actions": self.action_processor.json_to_action_vector(demo["actions"][i : i + 128]), # The next 128 actions in the situation
                        }
                    )

        return situations


    def create_situations(self, encoded_demos, window_size=128, stride=2):
        situations = []
        for demo in tqdm(encoded_demos, desc="Creating situations"):
            for i in range(
                window_size, len(demo["encoded_demo"]) - window_size, stride
            ):
                situations.append(
                    {
                        "demo_id": demo["demo_id"],
                        "sit_frame_idx": i, # Frame index of the situation in the video
                        "situation": demo["encoded_demo"][i],
                        "actions": self.action_processor.json_to_action_vector(demo["actions"][i : i + 128]), # The next 128 actions in the situation
                    }
                )
        return situations

    def situation_pipeline(self):
        demonstrations = self.load_demonstrations()
        encoded_demos = self.encode_demonstrations(demonstrations)
        situations = self.create_situations(encoded_demos)
        return situations

    def _load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return frames

    def _load_jsonl(self, jsonl_path):
        with open(jsonl_path) as f:
            return [json.loads(line) for line in f]

    def save_situations(self, situations, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(situations, f)

    def load_situations(self, save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)


class Memory:
    """FAISS based Memory class for indexing and retrieving situations"""

    def create_index(self, situations):
        self.index = faiss.IndexFlatL2(1024)
        self.index.add(self._create_situation_array(situations))

        self.situation_ids = [
            {"demo_id": x["demo_id"], "sit_frame_idx": x["sit_frame_idx"], "actions": x["actions"]}
            for x in situations
        ]

    def search(self, query, k=4):
        distances, nearest_indices = self.index.search(query.reshape(1, 1024), k)
        result = []
        for i, idx in enumerate(nearest_indices[0]):
            result.append(
                {   
                    "idx": int(idx),
                    "demo_id": self.situation_ids[idx]["demo_id"],
                    "sit_frame_idx": self.situation_ids[idx]["sit_frame_idx"], # Frame index of the situation in the video
                    "distance": distances[0][i],
                    "actions": self.situation_ids[idx]["actions"],
                }
            )
        return result

    def save_index(self, save_dir, filename="memory"):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, filename + ".index"))

        with open(os.path.join(save_dir, filename + ".json"), "w") as f:
            json.dump(
                {
                    "index_file": filename + ".index",
                    "situation_ids": self.situation_ids,
                },
                f,
            )

    def load_index(self, json_path):
        with open(json_path, "r") as f:
            situations_meta = json.load(f)

        self.situation_ids = situations_meta["situation_ids"]
        self.index = faiss.read_index(
            os.path.join(os.path.dirname(json_path), situations_meta["index_file"])
        )

    def _create_situation_array(self, situations):
        """Create numpy array of situation latents from situations dictionary"""
        situation_latents = np.array([x["situation"] for x in situations])
        return situation_latents
