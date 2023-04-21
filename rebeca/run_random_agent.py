from argparse import ArgumentParser
import pickle

import gym
import minerl

from openai_vpt.agent import MineRLAgent

def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=False):
    env = gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    env.seed(4006)
    env.reset()    

    for _ in range(n_episodes):
        env.seed(4006)
        obs = env.reset()
        for i in range(max_steps):
            # Step your model here.
            # Currently, it's doing no actions
            # for 200 steps before quitting the episode
            no_act = env.action_space.noop()

            if i < 200:
                no_act["ESC"] = 0
            else:
                no_act["ESC"] = 1

            obs, reward, done, info = env.step(no_act)

            if show:
                env.render()
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
