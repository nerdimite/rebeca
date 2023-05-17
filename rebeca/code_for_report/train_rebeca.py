data_loader = DataLoader(
    dataset_dir=data_dir,
    n_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    n_epochs=EPOCHS,
    num_demonstrations=NUM_DEMOS
)

# Keep track of the hidden state per episode/trajectory.
# DataLoader provides unique id for each episode, which will
# be different even for the same trajectory when it is loaded
# up again
episode_hidden_states = {}

for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
    batch_loss = 0
    for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):

        agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
    
        if episode_id not in episode_hidden_states:
            episode_hidden_states[episode_id] = agent.controller.initial_state(1)
        agent_state = episode_hidden_states[episode_id]

        action_logits, new_agent_state = agent(image, agent_state)

        # Make sure we do not try to backprop through sequence
        new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
        episode_hidden_states[episode_id] = new_agent_state

        # Finally, update the agent to increase the probability of the
        # taken action.
        log_prob  = agent.get_logprob_of_action(action_logits, agent_action)

        # Finally, update the agent to increase the probability of the
        # taken action.
        # Remember to take mean over batch losses
        loss = -log_prob / BATCH_SIZE
        batch_loss += loss.item()
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()