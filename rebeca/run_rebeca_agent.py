from argparse import ArgumentParser
import pickle

import gym
import minerl
import torch
import cv2
from time import sleep

from rebeca import REBECA
from seeds import SEEDS

import logging
import coloredlogs
coloredlogs.install(logging.DEBUG)

def main(cnn_model, trf_model, cnn_weights, trf_weights, memory, env, seed=0, max_steps=int(1e9), show=True):
    env = gym.make(env)
    rebeca = REBECA(cnn_model, trf_model, cnn_weights, trf_weights, memory, device='cuda').to('cuda')
    rebeca.to(rebeca.device)
    print('Model loaded')

    cap = cv2.VideoCapture(SEEDS[seed])
    # obs = env.reset()
    state = rebeca.controller.initial_state(1)

    with torch.inference_mode():
        for _ in range(max_steps):
            ret, curr_obs = cap.read()
            if not ret:
                break
            
            action, state_out = rebeca(curr_obs, state)
            # obs, _, done, _ = env.step(action)
            
            if show:
                cv2.imshow('MineRL', curr_obs)
                cv2.waitKey(1)
                # env.render()

    cv2.destroyAllWindows()
    cap.release()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--cnn-model", default="data/VPT-models/foundation-model-1x.model", type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--trf-model", default="data/VPT-models/foundation-model-1x.model", type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--cnn-weights", default="data/VPT-models/foundation-model-1x-cnn.weights", type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--trf-weights", default="data/VPT-models/foundation-model-1x-trf.weights", type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--memory", type=str, default="data/memory_cnn.json", help="Path to the memory '.json' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLBasaltMakeWaterfall-v0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    main(args.cnn_model, args.trf_model, args.cnn_weights, args.trf_weights, args.memory, args.env, args.seed)
