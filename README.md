# REBECA (Retrieval Enhanced Behavioral Cloned Agent)

## Abstract

Autonomous agents in open-ended environments, such as Minecraft, self-driving cars, and robotic manipulation, are commonly trained using behavioral cloning and imitation learning. However, these approaches are known to suffer from limitations, including dataset bias, overfitting, and the inability to dynamically adapt or generalize, as well as a lack of causal modeling. In this work, we propose Retrieval Enhanced Behavioral Cloned Agent (REBECA), which dynamically retrieves expert trajectories similar to the agent's current observation from a database and uses a controller neural network to select appropriate actions in the environment. Our approach uses a frozen Video PreTrained (VPT) model as an encoder for the retrieval system and a transformers-based network as a controller. We evaluate REBECA on the MineRL Basalt tasks, an open-ended sandbox environment that demonstrates real-world challenges. Our research objective is to surpass a vanilla behavioral cloned agent while also evaluating the method's ability to adapt to new tasks dynamically by simply swapping the trajectory database. Our proposed method has potential applications in various real-world use cases, including autonomous driving, robotic manipulation tasks, few-shot learning, and any open-ended environment that requires sequential decision making.

## Task

Our model is evaluated using the MineRL BASALT MakeWaterfall task, which involves spawning in a mountainous region with a water bucket and a variety of tools, constructing a stunning waterfall, and then relocating to capture a scenic photograph of the waterfall by pressing the ESCAPE key. The episode concludes when the ESCAPE key is pressed.
