import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from torch import nn
from model import VPTEncoder, Controller
from memory import Memory

import torch as th
from gym3.types import DictType
from gym import spaces

from openai_vpt.lib.action_mapping import CameraHierarchicalMapping
from openai_vpt.lib.actions import ActionTransformer
from openai_vpt.lib.policy import MinecraftAgentPolicy
from openai_vpt.lib.torch_util import default_device_type, set_default_torch_device

def l2_distance(a, b):
    return euclidean(a, b) ** 2

# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


def l2_distance(a, b):
    return euclidean(a, b) ** 2

class Retriever:
    def __init__(self, encoder_model, encoder_weights, memory_path):
        self.vpt = VPTEncoder(encoder_model, encoder_weights)
        self.vpt.eval()
        self.memory = Memory()
        self.memory.load_index(memory_path)

        self.reset()

    def encode_query(self, query_obs):
        query_obs_vec, state_out = self.vpt(query_obs, self.hidden_state)
        self.hidden_state = state_out
        query_obs_vec = query_obs_vec.squeeze().cpu().numpy()
        return query_obs_vec

    def retrieve(self, query_obs, k=2, encode_obs=True):
        if encode_obs:
            query_obs = self.encode_query(query_obs)
        results = self.memory.search(query_obs, k=k)

        if (
            results[0]["distance"] == 0
        ):  # to prevent returning the same situation and overfitting
            print("Same situation found")
            return results[1], query_obs
        else:
            return results[0], query_obs

    def reset(self):
        self.hidden_state = self.vpt.policy.initial_state(1)

class REBECA(nn.Module):
    def __init__(self, encoder_model, encoder_weights, memory_path, device="auto"):
        super().__init__()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.retriever = Retriever(encoder_model, encoder_weights, memory_path)

        # Unfreeze final layers
        self.vpt = VPTEncoder(encoder_model, encoder_weights, freeze=True)
        self.trainable_parameters = []
        for param in self.vpt.policy.lastlayer.parameters():
            param.requires_grad = True
            self.trainable_parameters.append(param)

        self.controller = Controller()

        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def forward(self, obs, env):
        # retrieve situations only if the agent diverges from the previous situation
        if self.current_situation is None:
            result, query_obs_vec = self.retriever.retrieve(obs)
            self.current_situation = result
            self.situation_counter += 1
        else:
            query_obs_vec = self.retriever.encode_query(obs)
            if (
                l2_distance(
                    query_obs_vec,
                    self.retriever.memory.index.reconstruct(self.current_situation["idx"]),
                )
                > 300
                or self.situation_counter > 128
            ):
                result, query_obs_vec = self.retriever.retrieve(
                    query_obs_vec, encode_obs=False
                )
                self.current_situation = result
                self.situation_counter = 0
            else:
                self.situation_counter += 1
                result = self.current_situation

        situation = torch.Tensor(
            self.retriever.memory.index.reconstruct(result["idx"])
        ).reshape(1, 1, -1)
        obs_vector, self.hidden_state = self.vpt(obs, self.hidden_state)

        retrieved_actions = {
            "camera": self._one_hot_encode(result["actions"]["camera"], 121).to(
                self.device
            ),
            "keyboard": self._one_hot_encode(result["actions"]["buttons"], 8641).to(
                self.device
            ),
        }

        action = self.controller(
            obs_vector.to(self.device), situation.to(self.device), retrieved_actions
        )

        return env.action_space.noop()

    def reset(self):
        self.hidden_state = self.vpt.policy.initial_state(1)
        self.current_situation = None
        self.situation_counter = 0
        self.retriever.reset()

    def _one_hot_encode(
        self, actions: list, num_classes: int, add_batch_dim: bool = True
    ):
        """One-hot encodes the actions"""
        actions = torch.tensor(actions)
        if add_batch_dim:
            actions = actions.unsqueeze(0)
        return torch.nn.functional.one_hot(actions, num_classes=num_classes).float()

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: torch.from_numpy(v).to(self.device) for k, v in action.items()}
        return action
