from argparse import ArgumentParser
import pickle
import json
import numpy as np
import gym
import minerl

from openai_vpt.agent import MineRLAgent

def main(model, env, n_episodes=1, show=False):
    env = gym.make(env)
    # agent_parameters = pickle.load(open(model, "rb"))
    # policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    # pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    # pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    # agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    # agent.load_weights(weights)

    seed = 4006
    env.seed(seed)
    env.reset()

    # Load expert actions
    with open(f"eval_seeds/demo-{seed}.json", "r") as f:
        expert_actions = json.load(f)

    for _ in range(n_episodes):
        env.seed(seed)
        obs = env.reset()
        for act in expert_actions:
            # Step your model here.
            # Currently, it's doing no actions
            # for 200 steps before quitting the episode
            no_act = env.action_space.noop()

            obs, reward, done, info = env.step(act)

            if show:
                env.render()
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    # parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="data/VPT-models/foundation-model-1x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLBasaltMakeWaterfall-v0")
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.env, show=args.show)
