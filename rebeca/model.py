import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
import torch

from openai_vpt.lib.policy import MinecraftPolicy
from openai_vpt.lib.tree_util import tree_map

class VPTEncoder(torch.nn.Module):
    """VPT Encoder for Embedding Situations and Single Observations"""

    def __init__(self, model_path, weights_path=None, freeze=True, device="auto"):
        super().__init__()

        agent_policy_kwargs = self.load_model_parameters(model_path)
        self.policy = MinecraftPolicy(**agent_policy_kwargs, single_output=True)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.policy.to(self.device)

        if weights_path is not None:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )

        if freeze:
            for param in self.policy.parameters():
                param.requires_grad = False

        self.dummy_first = (
            torch.from_numpy(np.array((False,))).to(self.device).unsqueeze(1)
        )

    def load_model_parameters(self, model_path):
        """Load model parameters from model_path"""

        with open(model_path, "rb") as f:
            agent_parameters = pickle.load(f)
            policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]

            return policy_kwargs

    def preprocess_obs(self, obs_frame):
        """Turn observation from MineRL environment into model's observation"""
        policy_input = cv2.resize(
            obs_frame, (128, 128), interpolation=cv2.INTER_LINEAR
        )[None]
        policy_input = {"img": torch.from_numpy(policy_input).to(self.device)}
        policy_input = tree_map(lambda x: x.unsqueeze(1), policy_input)
        return policy_input

    def forward(self, obs, state_in):
        """Encode observation into latent space"""

        obs = self.preprocess_obs(obs)
        latent_vec, state_out = self.policy(
            obs, state_in, context={"first": self.dummy_first}
        )

        return latent_vec, state_out

    def encode_trajectory(self, trajectory):
        """Encode expert trajectory frames into a latent vector with state history"""

        with torch.inference_mode():
            initial_state = self.policy.initial_state(1)
            hidden_state = initial_state
            latent_vectors = []

            for obs in tqdm(trajectory, desc="Encoding Trajectory", leave=False):
                latent, state_out = self(obs, hidden_state)
                hidden_state = state_out
                latent_vectors.append(latent.squeeze().detach().cpu().numpy())

            return latent_vectors