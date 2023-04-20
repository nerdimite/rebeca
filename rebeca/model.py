import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from torch import nn

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

    def encode_trajectory(self, trajectory, tolist=False):
        """Encode expert trajectory frames into a latent vector with state history"""

        with torch.inference_mode():
            initial_state = self.policy.initial_state(1)
            hidden_state = initial_state
            latent_vectors = []

            for obs in tqdm(trajectory, desc="Encoding Trajectory", leave=False):
                latent, state_out = self(obs, hidden_state)
                hidden_state = state_out
                latent = latent.squeeze().detach().cpu().numpy()
                if tolist:
                    latent = latent.tolist()
                latent_vectors.append(latent)

            return latent_vectors


class Controller(nn.Module):
    """Applies Cross Attention on the observation embedding with the situation embeddings and the next 128 actions"""

    def __init__(self) -> None:
        super().__init__()

        # Define some constants
        d_model = 1024  # dimension of the input and output vectors
        d_k = 1024  # dimension of the query and key vectors
        d_v = 1024  # dimension of the value vectors
        n_heads = 4  # number of attention heads
        assert d_model % n_heads == 0  # make sure d_model is divisible by n_heads

        # Define linear layers for projection
        self.Wq_obs = nn.Linear(
            d_model, d_k
        )  # project observation embedding to query vector
        self.Wk_sit = nn.Linear(
            d_model, d_k
        )  # project situation embedding to key vector
        self.Wv_sit = nn.Linear(
            d_model, d_v
        )  # project situation embedding to value vector
        self.Wk_key = nn.Linear(
            8641, d_k
        )  # project keyboard action one-hot vector to key vector
        self.Wv_key = nn.Linear(
            8641, d_v
        )  # project keyboard action one-hot vector to value vector
        self.Wk_cam = nn.Linear(
            121, d_k
        )  # project camera action one-hot vector to key vector
        self.Wv_cam = nn.Linear(
            121, d_v
        )  # project camera action one-hot vector to value vector

        # Define multi-head attention layer
        self.attention = nn.MultiheadAttention(d_model, n_heads)

        # TODO: Dynamic scaling factor for cross-attention importance
        # TODO: condition on distance between observation and situation
        # self.alpha_head = nn.Linear(d_model, 1)
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.alpha = 1.0

        # Define output layer for concatenation or addition
        self.Wo = nn.Linear(d_v, d_model)  # project output vector to original dimension

        self.action_head = nn.ModuleDict(
            {"camera": nn.Linear(d_model, 121), "keyboard": nn.Linear(d_model, 8641)}
        )

    def forward(self, observation, situation, actions):
        # Project input tensors to query, key and value vectors
        q_obs = self.Wq_obs(observation)  # query vector tensor of shape [1, 1, 1024]
        k_sit = self.Wk_sit(situation)  # key vector tensor of shape [1, 1, 1024]
        v_sit = self.Wv_sit(situation)  # value vector tensor of shape [1, 1, 1024]
        k_key = self.Wk_key(
            actions["keyboard"]
        )  # key vector tensor of shape [1, 128, 1024]
        v_key = self.Wv_key(
            actions["keyboard"]
        )  # value vector tensor of shape [1, 128, 1024]
        k_cam = self.Wk_cam(
            actions["camera"]
        )  # key vector tensor of shape [1, 128, 1024]
        v_cam = self.Wv_cam(
            actions["camera"]
        )  # value vector tensor of shape [1, 128, 1024]

        # Concatenate all the key and value vectors along the second dimension
        k_all = torch.cat(
            [k_sit, k_key, k_cam], dim=1
        )  # key vector tensor of shape [1, 257, 1024]
        v_all = torch.cat(
            [v_sit, v_key, v_cam], dim=1
        )  # value vector tensor of shape [1, 257, 1024]

        # Apply multi-head attention on the query and key-value pairs
        out_obs, _ = self.attention(q_obs, k_all.transpose(0, 1), v_all.transpose(0, 1))

        # Add the output vector with the original query vector
        # alpha = torch.sigmoid(self.alpha_head(observation)).squeeze()
        out_obs = q_obs + (out_obs * self.alpha)

        # Apply output layer on the output vector
        out_obs = self.Wo(out_obs)

        # Apply action head on the output vector
        out_key = self.action_head["keyboard"](out_obs)
        out_cam = self.action_head["camera"](out_obs)

        return out_key.reshape(1, -1), out_cam.reshape(1, -1)

