import gym
import numpy as np
# Import matplotlib to plot the outcomes
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# Initialize the non-slippery Frozen Lake environment
environment = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 500        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training
for _ in range(episodes):
    state = environment.reset()
    done = False
    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()
        print("rnd:", rnd)
        # If random number < epsilon, take a random action
        if rnd < epsilon:
          action = environment.action_space.sample()

        # Else, take the action with the highest value in the current state
        else:
          action = np.argmax(qtable[state[0]])
             
        # Implement this action and move the agent in the desired direction
        result = environment.step(action)
        new_state, reward, done = result[0], result[1], result[2]
        # Update Q(s,a)
        qtable[state[0], action] = qtable[state[0], action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state[0], action])
        
        # Update our current state
        state = [new_state, 1]
        print(state)
        print(qtable)
        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"
    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)
    print("epsilon:", epsilon)

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
plt.bar(range(len(outcomes)), outcomes, width=1.0)
plt.show()