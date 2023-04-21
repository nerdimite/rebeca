'''
This script is used to make predictions on a video using the inverse dynamics model.

python predict_inverse_dynamics_model.py \
--weights models/4x_idm.weights \
--model models/4x_idm.model \
--video-path data/cheeky-cornflower-setter-0a060253331a-20220717-155545.mp4
'''

from argparse import ArgumentParser
from tqdm.auto import tqdm
import pickle
import cv2
import numpy as np
import json
import torch as th

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MESSAGE = """
This script will take a video, predict env actions for its frames and save it as a json file.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def convert_idm_actions_env_actions(batch_predicted_actions):
    '''Convert the batch of IDM actions to list of env actions'''

    batch_size = batch_predicted_actions['attack'].shape[1]
    env_actions = [NOOP_ACTION.copy() for _ in range(batch_size)]

    for action, action_array in batch_predicted_actions.items():
        action_array = action_array.squeeze(0)
        
        for i in range(len(action_array)):
            if action == 'camera':
                env_actions[i][action] = action_array[i].tolist()
            elif action_array[i] == 1:
                env_actions[i][action] = 1
    
    return env_actions


def main(model, weights, video_path, batch_size):
    print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    required_resolution = ENV_KWARGS["resolution"]
    print(f'Resolution: {required_resolution}')
    cap = cv2.VideoCapture(video_path)

    # Load frames in batches
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
        # BGR -> RGB
        frames.append(frame[..., ::-1])
    
    # IDM Predictions
    predicted_env_actions = []
    for i in tqdm(range(0, len(frames), batch_size)):
        th.cuda.empty_cache()
        predicted_actions = agent.predict_actions(np.array(frames[i:i+batch_size]))
        pred_env_actions = convert_idm_actions_env_actions(predicted_actions)
        predicted_env_actions.extend(pred_env_actions)


    # Save the jsonl file
    actions_path = video_path.replace(".mp4", ".json") 
    with open(actions_path, "w") as f:
        json.dump(predicted_env_actions, f)


if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of frames to process at a time.")

    args = parser.parse_args()

    main(args.model, args.weights, args.video_path, args.batch_size)
