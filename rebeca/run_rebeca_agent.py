from argparse import ArgumentParser
import pickle

import gym
import minerl
import cv2
from time import sleep

# from rebeca import REBECA
from seeds import SEEDS

import logging
import coloredlogs
coloredlogs.install(logging.DEBUG)

def main(model, weights, memory, env, seed=0, max_steps=int(1e9), show=True):
    env = gym.make(env)
    # rebeca = REBECA(model, weights, memory, device='cuda')
    # rebeca.to(rebeca.device)
    print('Model loaded')

    cap = cv2.VideoCapture(SEEDS[seed])
    # obs = env.reset()

    for _ in range(max_steps):
        ret, obs = cap.read()
        # action = rebeca(obs)
        action = env.action_space.noop()
        
        # obs, _, done, _ = env.step(action)
        
        if not ret:
            break
        if show:
            cv2.imshow('MineRL', obs)
            cv2.waitKey(8)
            # env.render()

    cv2.destroyAllWindows()
    cap.release()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, default="data/VPT-models/foundation-model-1x-net.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="data/VPT-models/foundation-model-1x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--memory", type=str, default="data/memory.json", help="Path to the memory '.json' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLBasaltMakeWaterfall-v0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    main(args.model, args.weights, args.memory, args.env, args.seed)
