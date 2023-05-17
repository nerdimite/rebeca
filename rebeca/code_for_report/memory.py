import os
import json
from tqdm.auto import tqdm
import numpy as np
import faiss


class Memory:
    """FAISS based Memory class for indexing and retrieving situations"""

    def create_index(self, situations):
        self.index = faiss.IndexFlatL2(1024)
        self.index.add(self._create_situation_array(situations))

        # Store the situations without the situation embeddings
        self.situations_meta = [{k: v for k, v in s.items() if k != "situation"} for s in situations]

    def search(self, query, k=4):
        distances, nearest_indices = self.index.search(query.reshape(1, 1024), k)
        result = []
        for i, idx in enumerate(nearest_indices[0]):
            result.append(
                {   
                    "idx": int(idx),
                    "distance": distances[0][i],
                    **self.situations_meta[idx],
                    'embedding': self.index.reconstruct(int(idx))
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
                    "situations_meta": self.situations_meta,
                },
                f,
            )

    def load_index(self, json_path):
        with open(json_path, "r") as f:
            situations_meta = json.load(f)

        self.situations_meta = situations_meta["situations_meta"]
        self.index = faiss.read_index(
            os.path.join(os.path.dirname(json_path), situations_meta["index_file"])
        )

    def _create_situation_array(self, situations):
        """Create numpy array of situation latents from situations dictionary"""
        situation_latents = np.array([x["situation"] for x in situations])
        return situation_latents
