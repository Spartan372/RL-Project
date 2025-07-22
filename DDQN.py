import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import time
from ReplayBuffer import ReplayMemory

# Neural Networks
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_dense):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_dense)
        self.fc2 = nn.Linear(num_dense, num_dense)
        self.fc3 = nn.Linear(num_dense, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Methods
class DDQNAgent:
    def __init__(self,                      
                 env_name = "CartPole-v1",      # Gym Environment specification
                 learning_rate = 0.001,         # Learning Rate - Size of table updates
                 gamma = 0.99,                  # How important future values are to the Q-Value
                 epsilon = 1.0,                 # Exploration rate where 1 = 100% 
                 epsilon_min = 0.01,            # Minimum value epsilon can reach
                 epsilon_decay = 0.995,         # Reduction rate of epsilon per episode
                 batch_size = 64,               # Number of states sampled from replay memory for training
                 memory_size = 10000,           # Number of experiences stored in replay memory 
                 target_update_freq = 1000,     # How often target network is updated from the main network
                 num_dense = 128):              # Number of neurons in the dense layer of the Q-Network
    
        # Gather Env Dimensions
        self.env = gym.make(env_name)
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n

        # Initialise Neural Networks
        self.policy_net = NeuralNetwork(input_dim, output_dim, num_dense)
        self.target_net = NeuralNetwork(input_dim, output_dim, num_dense)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(memory_size)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        #Add Hyperparameters to the Object
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.steps_done = 0

    # Choose action either randomly or through Deep Q Network (Exploration vs Exploitation)
    def _select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

    # Train model based on replay memory sampling if called
    def optimise_model(self):

        # Only run if enough samples exist in replay memory
        if len(self.memory) < self.batch_size:
            return

        # Group state dimensions into tensors 
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # Model predicts q values for each Action
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # For each sample
        with torch.no_grad():
            # Select the max value action based on the policy network
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)

            # Evaluate the chosen action using the target_net network
            max_next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()

            # Calculate Q-Value based on Bellman's equation for learning
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)


        # Calc the Mean Squared Error of the predicted Q_values (q-values) and the actual q-values(target_q_values)
        # Optimise the model based on the MSE
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    # Gather Samples, train model, and return episodes performance history
    def train(self, num_episodes=500):

        # Initialise multi-episode variables
        total_time = 0
        total_reward = 0
        History = []

        # Episode Loop
        for episode in range(num_episodes):
            initial_time = time.time()   #set the initial time
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self._select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                self.optimise_model()

                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.steps_done += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_time = time.time() - initial_time #episode total time
            total_time = total_time + episode_time
            total_reward += episode_reward
            History.append((episode, episode_reward, episode_time))

            #At each 100 episodes, Print the Average Time and Average Score
            if episode % 100 == 0: 
                print(f"[Episode {episode}] Time: {total_time / 100:.4f} s | Mean Reward: {total_reward / 100:.2f} | Epsilon: {self.epsilon:.4f}")
                total_time = 0
                total_reward = 0

            
        return History

    # Run the model without updating value
    def test(self, attempts=10, render=True):
        total_reward = 0        
        self.policy_net.eval()  # Sets model to evaluation mode

        # Determine render mode
        if render:
            env = gym.make("CartPole-v1", render_mode = 'human') 
        else:
            env = gym.make("CartPole-v1") 

        # Attempt Loop
        for attempt in range(1, attempts + 1):

            initial_time = time.time()                  #set the initial time
            state, _ = env.reset()                      #Grab the raw state 
            episode_reward = 0  

            #Environment Loop
            done = False
            while not done:

                #Choose Action based on model prediction
                action = torch.argmax(self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))).item()
                
                #Play out the Action
                new_state, action_reward, terminated, truncated, _ = env.step(action)       #Step action and receive data
                done = terminated or truncated                                              #Merge end variables into one collective variable
                episode_reward += action_reward                                                     #Add step reward to total reward

                state = new_state

            final_time = time.time() - initial_time  
            total_reward += episode_reward          
            print(f"[Test {attempt}] Time: {final_time:.2f} s | Score: {episode_reward:.2f}")
        
        env.close()

        avg_reward = total_reward / attempts
        print(f"[Test Summary] Average Score: {avg_reward:.2f}")
        return avg_reward
    