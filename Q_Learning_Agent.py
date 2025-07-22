import gymnasium as gym
import numpy as np
import time 

class QLearningAgent:
    def __init__(self, 
                observation_quantisation = [30, 30, 50, 50],                # Number of discrete values for each state dimension
                observation_intervals = np.array([0.25, 0.25, 0.01, 0.1]),  # Value gap for each discrete value 
                env_name = "CartPole-v1",                                   # Gym Environment specification
                alpha = 0.9,                                                # Learning Rate - Size of table updates
                gamma = 0.95,                                               # How important future values are to the Q-Value
                epsilon = 1,                                                # Exploration rate where 1 = 100% 
                epsilon_decay = 0.9997,                                     # Reduction rate of epsilon per episode
                min_epsilon = 0.1):                                         # Minimum value epsilon can reach
        
        self.env = gym.make(env_name)                               
        self.observation_quantisation  = observation_quantisation     # Number of possible values for each observation
        self.observation_intervals = observation_intervals          # Size difference between each possible observation value 
        self.env.close()

        #Q-Value Table
        self.q_table = np.random.uniform(low=0, high=1, size=(self.observation_quantisation + [self.env.action_space.n])) #Create table for Q-Learning

        #Add Hyperparameters to the Object
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
    
    #Rounds each state dimension to the closest discrete value
    def _get_discrete_state(self, state,  offset = np.array([15,10,1,10])): 
        discrete_state = state/self.observation_intervals + offset  #Divide each dimension by a value, then add the offset (Avoids negative values)
        return tuple(discrete_state.astype(int))

    #Explore Environment and update Q-Value table accordingly
    def train(self, num_episodes = 10000):
        env = gym.make("CartPole-v1") #Open-AI Environment
        total_time = 0
        total_reward = 0

        History = []

        #Episode Loop
        for episode in range(0, num_episodes):
            
            #Episode level variables
            initial_time = time.time()                          #set the initial time
            initial_state, _ = env.reset()                      #Grab the raw state 
            state = self._get_discrete_state(initial_state)     #Round the observation into quantised state
            episode_reward = 0                                  #reset reward to 0

            #Attempt Loop
            done = False #Reset done variable for each attempt
            while not done: 

                #Choose Action based on the epsilon greedy strategy
                if np.random.random() > self.epsilon:               #Choose best action if random > epsilon
                    action = np.argmax(self.q_table[state])         #Take action based on Q-Table
                else:
                    action = np.random.randint(0, env.action_space.n) #Choose a Random Action

                #Play out the Action
                new_state, action_reward, terminated, truncated, _ = env.step(action)   #Step action and receive data
                done = terminated or truncated                                          #Merge end variables into one collective variable
                episode_reward += action_reward                                         #Add step reward to total reward
                new_state = self._get_discrete_state(new_state)                         #Quantise the state

                #update q-table
                if not done: 
                    Q_Future = np.max(self.q_table[new_state])      # Estimate max future reward (greedy)
                    Q_Current = self.q_table[state + (action,)]     # Get the Current Q Value for State, Action
                    Q_New = (1 - self.alpha) * Q_Current + self.alpha * (action_reward + self.gamma * Q_Future) #Bellman's Equation
                    self.q_table[state + (action,)] = Q_New

                state = new_state
            
            #Update and print epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            episode_time = time.time() - initial_time 
            total_time = total_time + episode_time #Add episode time to the time taken for all episodes so far

            total_reward += episode_reward # Add the episode reward to the cumulative reward for all episodes so far

            #At each 500 episodes, Print the Average Time and Average Score
            if episode % 500 == 0: 
                print(f"[Episode {episode}] Time: {total_time / 500:.4f} s | Mean Reward: {total_reward / 500:.2f} | Epsilon: {self.epsilon:.4f}")
                total_time = 0
                total_reward = 0

            #Add episode result to episode history
            History.append((episode, episode_reward, episode_time))

        env.close()
        return History

    # Run the model without updating value
    def test(self, attempts = 10, render = True):

        total_reward = 0
        
        # Determine render mode
        if render:
            env = gym.make("CartPole-v1", render_mode = 'human') 
        else:
            env = gym.make("CartPole-v1") 

        #Attempt Loop
        for attempt in range(1, attempts + 1):

            initial_time = time.time()                          #set the initial time
            initial_state, _ = env.reset()                      #Grab the raw state 
            state = self._get_discrete_state(initial_state)     #Round the observation into refined state
            episode_reward = 0                                  #Initialise reward to 0

            #Environment Loop
            done = False #Reset done variable for each attempt
            while not done: 

                #Choose Action based on model prediction
                action = np.argmax(self.q_table[state]) #Take action based on Q-Table

                #Play out the Action
                new_state, action_reward, terminated, truncated, _ = env.step(action)       #Step action and receive data
                done = terminated or truncated                                              #Merge end variables into one collective variable
                episode_reward += action_reward                                             #Add step reward to total reward
                new_state = self._get_discrete_state(new_state)                             #Refine the state

                state = new_state
            

            final_time = time.time() - initial_time  
            total_reward += episode_reward          
            print(f"[Test {attempt}] Time: {final_time:.2f} s | Score: {episode_reward:.2f}")



        env.close()
        avg_reward = total_reward / attempts
        print(f"[Test Summary] Average Score: {avg_reward:.2f}")
        return avg_reward


