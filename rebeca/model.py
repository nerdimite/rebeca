import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from torch import nn

from openai_vpt.lib.policy import MinecraftPolicy, VPTCNN, VPTRecurrence
from openai_vpt.lib.tree_util import tree_map

class VPTCNNEncoder(torch.nn.Module):
    """VPT CNN Encoder for Embedding Situations and Single Observations"""

    def __init__(self, model_path, weights_path=None, freeze=True, device="auto"):
        super().__init__()

        agent_policy_kwargs = self.load_model_parameters(model_path)
        self.policy = VPTCNN(**agent_policy_kwargs)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.policy.to(self.device)

        if weights_path is not None:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location=self.device),
                strict=False,
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

    def forward(self, obs):
        """Encode observation into latent space"""

        obs = self.preprocess_obs(obs)
        latent_vec = self.policy(obs)

        return latent_vec

    def encode_trajectory(self, trajectory, tolist=False):
        """Encode expert trajectory frames into a latent vector with state history"""

        with torch.inference_mode():
            latent_vectors = []

            for obs in tqdm(trajectory, desc="Encoding Trajectory", leave=False):
                latent = self(obs)
                latent = latent.squeeze().detach().cpu().numpy()
                if tolist:
                    latent = latent.tolist()
                latent_vectors.append(latent)

            return latent_vectors

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


class IntraSituationCA(nn.Module):
    '''Intra-Situation Cross Attention'''
    def __init__(self):
        super().__init__()

        # Define some constants
        d_model = 1024  # dimension of the input and output vectors
        d_k = 1024  # dimension of the query and key vectors
        d_v = 1024  # dimension of the value vectors
        n_heads = 4  # number of attention heads
        assert d_model % n_heads == 0  # make sure d_model is divisible by n_heads

        # Define linear layers for projection
        self.Wq_sit = nn.Linear(1024, d_model)  # project situation embedding to query vector
        self.Wk_key = nn.Linear(8641, d_k)  # project keyboard action one-hot vector to key vector
        self.Wv_key = nn.Linear(8641, d_v)  # project keyboard action one-hot vector to value vector
        self.Wk_cam = nn.Linear(121, d_k)  # project camera action one-hot vector to key vector
        self.Wv_cam = nn.Linear(121, d_v)  # project camera action one-hot vector to value vector
        
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, situation, actions):

        # Project input tensors to query, key and value vectors
        q_sit = self.Wq_sit(situation)  # query vector tensor of shape [1, 1, 1024]
        k_key = self.Wk_key(actions['buttons'])  # key vector tensor of shape [1, 20, 1024]
        v_key = self.Wv_key(actions['buttons'])  # value vector tensor of shape [1, 20, 1024]
        k_cam = self.Wk_cam(actions['camera'])  # key vector tensor of shape [1, 20, 1024]
        v_cam = self.Wv_cam(actions['camera'])  # value vector tensor of shape [1, 20, 1024]

        # Concatenate all the key and value vectors along the second dimension
        k_all = torch.cat([k_key, k_cam], dim=1)  # key vector tensor of shape [1, 40, 1024]
        v_all = torch.cat([v_key, v_cam], dim=1)  # value vector tensor of shape [1, 40, 1024]

        # Apply multi-head attention on the query and key-value pairs
        attn_out, _ = self.cross_attention(q_sit, k_all.transpose(0, 1), v_all.transpose(0, 1))

        return attn_out.transpose(0, 1)

class Controller(nn.Module):
    """Applies Cross Attention on the observation embedding with the situation embeddings and the next action"""

    def __init__(self, vpt_model, device='cuda'):
        super().__init__()

        # Define some constants
        d_model = 1024  # dimension of the input and output vectors
        d_k = 1024  # dimension of the query and key vectors
        d_v = 1024  # dimension of the value vectors
        n_heads = 4  # number of attention heads
        assert d_model % n_heads == 0  # make sure d_model is divisible by n_heads

        self.intra_situation_ca = IntraSituationCA()

        # Define linear layers for projection
        self.Wq_obs = nn.Linear(1024, d_model)  # project observation embedding to query vector
        self.Wk_sit = nn.Linear(1024, d_k)  # project situation embedding to key vector
        self.Wv_sit = nn.Linear(1024, d_v)  # project situation embedding to value vector
        self.Wk_next_key = nn.Linear(8641, d_k)  # project next keyboard action one-hot vector to key vector
        self.Wv_next_key = nn.Linear(8641, d_v)  # project next keyboard action one-hot vector to value vector
        self.Wk_next_cam = nn.Linear(121, d_k)  # project next camera action one-hot vector to key vector
        self.Wv_next_cam = nn.Linear(121, d_v)  # project next camera action one-hot vector to value vector

        # Define multi-head attention layer
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads)

        # TODO: Dynamic scaling factor for cross-attention importance
        # TODO: condition on distance between observation and situation
        # self.alpha_head = nn.Linear(d_model, 1)
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.alpha = 1.0

        # Define output layer for concatenation or addition
        self.Wo = nn.Linear(d_v, d_model)  # project output vector to original dimension

        # Define VPT Transformer layers
        self.vpt_transformers = VPTRecurrence(**self.load_vpt_parameters(vpt_model))
        self.dummy_first = torch.from_numpy(np.array((False,))).unsqueeze(1)

        # Define Action Heads
        self.action_head = nn.ModuleDict(
            {"camera": nn.Linear(d_model, 121), "buttons": nn.Linear(d_model, 8641)}
        )

    def forward(self, observation, situation, situation_actions, next_action, state_in):
        
        # Apply intra-situation cross attention
        situation = self.intra_situation_ca(situation, situation_actions)

        # Project input tensors to query, key and value vectors
        q_obs = self.Wq_obs(observation)  # query vector tensor of shape [1, 1, 1024]
        k_sit = self.Wk_sit(situation)  # key vector tensor of shape [1, 1, 1024]
        v_sit = self.Wv_sit(situation)  # value vector tensor of shape [1, 1, 1024]
        k_key = self.Wk_next_key(next_action['buttons'])  # key vector tensor of shape [1, 1, 1024]
        v_key = self.Wv_next_key(next_action['buttons'])  # value vector tensor of shape [1, 1, 1024]
        k_cam = self.Wk_next_cam(next_action['camera']) # key vector tensor of shape [1, 1, 1024]
        v_cam = self.Wv_next_cam(next_action['camera'])  # value vector tensor of shape [1, 1, 1024]

        # Concatenate all the key and value vectors along the second dimension
        # print(k_sit.shape, k_key.shape, k_cam.shape)
        # try:
        key_vec = torch.cat([k_sit, k_key, k_cam], dim=1)  # key vector tensor of shape [1, 3, 1024]
        val_vec = torch.cat([v_sit, v_key, v_cam], dim=1)  # value vector tensor of shape [1, 3, 1024]
        # except Exception as e:
        #     # print(k_sit.shape, k_key.shape, k_cam.shape)
        #     raise e

        # Apply multi-head attention on the query and key-value pairs
        out_obs, _ = self.cross_attention(q_obs, key_vec.transpose(0, 1), val_vec.transpose(0, 1))

        # Add the output vector with the original query vector
        # alpha = torch.sigmoid(self.alpha_head(observation)).squeeze()
        out_obs = q_obs + (out_obs * self.alpha)

        # Apply output layer on the output vector
        out_obs = self.Wo(out_obs)

        # Apply VPT Transformer
        latent, state_out = self.vpt_transformers(out_obs, state_in, self.dummy_first)

        # Apply action heads on the final latent vector
        out_key = self.action_head["buttons"](latent)
        out_cam = self.action_head["camera"](latent)

        return {
            'buttons': out_key.reshape(1, -1),
            'camera': out_cam.reshape(1, -1)
        }, state_out

    def initial_state(self, batch_size):
        return self.vpt_transformers.initial_state(batch_size)

    def load_vpt_parameters(self, model_path):
        """Load model parameters from model_path"""

        with open(model_path, "rb") as f:
            agent_parameters = pickle.load(f)
            policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]

            return policy_kwargs