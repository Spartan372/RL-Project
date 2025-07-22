import gymnasium as gym

from DQN import DQNAgent
from Q_Learning_Agent import QLearningAgent
from DDQN import DDQNAgent

'''
print('----------------Q-Learning----------------')
Balancing_Bob = QLearningAgent()
History = Balancing_Bob.train(num_episodes = 2500)
Average_Reward = Balancing_Bob.test(attempts = 5, render=False)
'''

'''
print('-------------------DQN--------------------')
agent = DQNAgent()
History = agent.train(num_episodes= 200)
Average_Reward = agent.test(attempts = 2, render=False)
'''

'''
print('------------------DDQN--------------------')
agent = DDQNAgent()
History = agent.train(num_episodes= 200)
Average_Reward = agent.test(attempts = 2, render=False)
'''