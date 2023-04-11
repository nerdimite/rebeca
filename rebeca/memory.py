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


class SituationLoader:
    """Data loader for loading expert demonstrations and creating situation embeddings"""

    def __init__(self, vpt_model: VPTEncoder, data_dir="data/MakeWaterfall/"):
        self.vpt = vpt_model
        self.load_expert_data(data_dir)

    def load_expert_data(self, data_dir):
        """Load expert demonstrations from data_dir"""

        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))

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
                {"demo_id": demo["demo_id"], "encoded_demo": encoded_demo}
            )
        return encoded_demos

    def create_situations(self, encoded_demos, window_size=128, stride=2):
        situations = []
        for demo in tqdm(encoded_demos, desc="Creating situations"):
            for i in range(
                window_size, len(demo["encoded_demo"]) - window_size, stride
            ):
                situations.append(
                    {
                        "demo_id": demo["demo_id"],
                        "situation_idx": i,
                        "situation": demo["encoded_demo"][i],
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
    """Memory class for indexing and retrieving situations"""

    def create_index(self, situations):
        self.index = faiss.IndexFlatL2(1024)
        self.index.add(self._create_situation_array(situations))

        self.situation_ids = [
            {"demo_id": x["demo_id"], "situation_idx": x["situation_idx"]}
            for x in situations
        ]

    def search(self, query, k=4):
        distances, nearest_indices = self.index.search(query.reshape(1, 1024), k)
        result = []
        for i, idx in enumerate(nearest_indices[0]):
            result.append(
                {
                    "demo_id": self.situation_ids[idx]["demo_id"],
                    "situation_idx": self.situation_ids[idx]["situation_idx"],
                    "distance": distances[0][i],
                    # TODO: add the next 128 actions in the situation
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

        self.situations_ids = situations_meta["situation_ids"]
        self.index = faiss.read_index(
            os.path.join(os.path.dirname(json_path), situations_meta["index_file"])
        )

    def _create_situation_array(self, situations):
        """Create numpy array of situation latents from situations dictionary"""
        situation_latents = np.array([x["situation"] for x in situations])
        return situation_latents
