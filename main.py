import gymnasium as gym
import numpy as np


env = gym.make("CartPole-v1", render_mode='human') #Environment and Mode Declaration
observation, info = env.reset(seed=123) #Initalise Variables
env.render #Start Window

episode_over = False

while not episode_over:
    action = env.action_space.sample() # Choose Random Action
    observation, reward, terminated, truncated, info = env.step(action) #Grab Details of the current state
    episode_over = terminated or truncated # Or Gate 


env.close() #Close Simulation