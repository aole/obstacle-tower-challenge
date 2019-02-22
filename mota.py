# Learning Obstacle Tower Agent (MOTA)

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

GAMMA       = 0.99
BATCH_SIZE  = 64
BATCH_EPOCH = 1
MEMORY_SIZE = 2024

ACTION_MAP = (
    [0,0,0,0], # frozen
    [1,0,0,0], # move forward
    [2,0,0,0], # move back
    [0,1,0,0], # camera right
    [0,2,0,0], # camera left
    [0,0,1,0], # jump
    [1,0,1,0]  # jump forward
    )
#ACTION_SIZE = 3*3*2*3
ACTION_SIZE = len(ACTION_MAP)

USE_ADVANTAGE = True

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm
    
class Model:
    def __init__(self):
        np.random.seed(4)
        # probabilities
        self.prob = [1]*ACTION_SIZE
        self.prob[1] = 2
        self.prob[5] = self.prob[6] = .2 # jump
        self.prob[3] = self.prob[4] = .5 # camera
        '''
        self.prob[0] = .1 # discourage standing still
        # reduce probability of jumping
        m = 0
        for i in range(3): #Forward/Back
            for j in range(3): #Camera
                for k in range(2):# Jump
                    for l in range(3):# left/right
                        if i==1 and j==k==l==0:
                            self.prob[m] += 150 # encourage moving forward
                            
                        m += 1
        '''
        self.prob = normalize(self.prob)
        #
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        
        self.memory = []
        
        self.model = Sequential()
    
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(168, 168, 3)))
        #self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        #self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(ACTION_SIZE, activation='linear'))

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse')
        
    def act(self, obs, explore=True):
        if explore and self.epsilon > self.epsilon_min and np.random.random()<self.epsilon:
            self.epsilon *= self.epsilon_decay
            return np.random.choice(ACTION_SIZE, 1, p=self.prob)[0]
            
        obs = np.reshape(obs[0], (-1, 168, 168, 3))
        out = self.model.predict(obs)
        action = np.argmax(out[0])
        
        return action #input_map[action]
        
    def remember(self, obs, action, reward, obsn, done):
        self.memory.append([obs[0], action, reward, obsn[0], done])
        if len(self.memory)>MEMORY_SIZE:
            del self.memory[0]
        
    def train(self, obs, action, reward, obsn, done):
        obs = np.reshape(obs[0], (-1, 168, 168, 3))
        obsn = np.reshape(obsn[0], (-1, 168, 168, 3))
        
        reward_predicted = self.model.predict(obs)
        Q_sa = self.model.predict(obsn)[0]
        
        reward_predicted[0][action] = reward
        if not done:
            reward_predicted[0][action] += GAMMA * np.max(Q_sa)
            # advantage
            if USE_ADVANTAGE:
                reward_predicted[0][action] -= np.mean(reward_predicted[0])
                
        self.model.fit(obs, reward_predicted, epochs=10, verbose=0)
        
    def train_batch(self):
        experience = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))

        obs, actions, rewards, obsn, done = zip(*experience)
        obs = np.asarray(obs)
        
        reward_predicted = self.model.predict(obs)
        Q_sa = self.model.predict(np.asarray(obsn))
        
        for i, ex in enumerate(experience):
            reward_predicted[i][actions[i]] = rewards[i]
            if not done[i]:
                reward_predicted[i][actions[i]] += GAMMA * np.max(Q_sa[i])
        self.model.fit(obs, reward_predicted, epochs=BATCH_EPOCH, verbose=0)
        
        
if __name__ == '__main__':
    model = Model()
    print('done!')