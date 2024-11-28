import gym
import random
import numpy as np

# Initialize the non-slippery Frozen Lake environment
environment = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
environment.reset()
environment.render()


qtable = np.zeros((16,4))
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n


print("qtable=")
print(qtable)

random.choice(["LEFT", "DOWN", "RIGHT", "UP"])

# 1. Randomly choose an action using action_space.sample()
"""
◀️ LEFT = 0
🔽 DOWN = 1
▶️ RIGHT = 2
🔼 UP = 3
"""
action = environment.action_space.sample()

# 2. Implement this action and move the agent in the desired direction
result = environment.step(action)
print(result)
new_state, reward, done = result[0], result[1], result[2]
# Display the results (reward and map)
environment.render()
print(f'Reward = {reward}')
