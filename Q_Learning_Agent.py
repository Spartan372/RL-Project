import gymnasium as gym
import numpy as np
import time 

class Q_Learning_Agent:
    def __init__(self, observation_levels = [30, 30, 50, 50], observation_intervals = np.array([0.25, 0.25, 0.01, 0.1])):
        env = gym.make("CartPole-v1") #Open-AI Environment
        self.observation_levels = observation_levels #Number of possible values for each observation
        self.observation_intervals = observation_intervals #Size differnce between each possible observation value 
        self.q_table = np.random.uniform(low=0, high=1, size=(self.observation_levels + [env.action_space.n])) #Create table for Q-Learning
        env.close()

    def get_discrete_state(self, state):
        offset = np.array([15,10,1,10])
        discrete_state = state/self.observation_intervals + offset
        return tuple(discrete_state.astype(int))

    def training(self, alpha = 0.9, gamma = 0.95, epsilon = 1, epsilon_decay = 0.9997, min_epsilon = 0.1, num_episodes = 10000, max_steps = 100):
        
        #Initalisation of variables
        env = gym.make("CartPole-v1") #Open-AI Environment
        total_time = 0
        total_reward = 0

        #Episode Loop
        for episode in range(num_episodes + 1): #Interate through episodes
            
            #Episode level variables
            inital_time = time.time()                       #set the initial time
            inital_state, _ = env.reset()              #Grab the unquantised state 
            state = self.get_discrete_state(inital_state)   #Round the observation into quantised state
            episode_reward = 0                              #reset reward to 0

            #Attempt Loop
            done = False #Reset done variable for each attempt
            while not done: 

                #Choose Action
                if np.random.random() > epsilon: #Randomly choose based on if random is larger then epsilon, 
                    action = np.argmax(self.q_table[state]) #Take action based on Q-Table
                else:
                    action = np.random.randint(0, env.action_space.n) #Do a Random Action

                #Play out the Action
                new_state, action_reward, win, loss, _ = env.step(action)  #Step action and recieve data
                done = win or loss                                          #Merge end variables into one collective variavble
                episode_reward += action_reward                                    #Add step reward to total reward
                new_state = self.get_discrete_state(new_state)              #Quantise the state

                if not done: #update q-table
                    Q_Future = np.max(self.q_table[new_state])      #Get the Best possible Q Value for the next State
                    Q_Current = self.q_table[state + (action,)]     #Get the Current Q Value for State, Action
                    Q_New = (1 - alpha) * Q_Current + alpha * (action_reward + gamma * Q_Future)
                    self.q_table[state + (action,)] = Q_New

                state = new_state
            
            #Update and print epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

            final_time = time.time() #episode has finished
            episode_time = final_time - inital_time #episode total time
            total_time = total_time + episode_time

            total_reward += episode_reward #episode total reward

            #At each 1000 episodes, Print the Average Time and Average Score
            if episode % 1000 == 0: 
                print("Time Average: " + str(total_time / 1000))
                total_time = 0

                print("Mean Reward: " + str(total_reward / 1000))
                total_reward = 0
            
            #Print Episode Milestone
            if episode % 2000 == 0: 
                print("Episode: " + str(episode))

        env.close()

    def Attempt(self, attempts = 1):
        
        env = gym.make("CartPole-v1", render_mode = 'human') #Open-AI Environment

        for attempt in range(1, attempts):

            inital_time = time.time()                       #set the initial time
            inital_state, _ = env.reset()              #Grab the unquantised state 
            state = self.get_discrete_state(inital_state)   #Round the observation into quantised state
            reward = 0                                      #Initalise reward to 0

            #Attempt Loop
            done = False #Reset done variable for each attempt
            while not done: 

                #Choose Action
                action = np.argmax(self.q_table[state]) #Take action based on Q-Table

                #Play out the Action
                new_state, action_reward, win, loss, _ = env.step(action)  #Step action and recieve data
                done = win or loss                                              #Merge end variables into one collective variavble
                reward += action_reward                                 #Add step reward to total reward
                new_state = self.get_discrete_state(new_state)                  #Quantise the state

                state = new_state

            env.render()

            #Print Attempt Number
            print("Attempt Number: ", attempt + 1)

            #Print Time Taken
            final_time = time.time() #episode has finished
            total_time = final_time - inital_time #episode total time
            print('Total Time: ', total_time)

            #Print Score Achieved
            print("Total Score: ", reward)

        env.close
        



