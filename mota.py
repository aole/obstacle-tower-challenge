# Learning Obstacle Tower Agent (MOTA)

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

GAMMA       = 0.99
BATCH_SIZE  = 32

class Model:
    def __init__(self):
        output_size = 3*3*2*3
        
        self.memory = []
        
        self.model = Sequential()
    
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(168, 168, 3)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(output_size, activation='linear'))

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse')
        
    def act(self, env, obs):
        #action = env.action_space.sample()
        
        obs = np.reshape(obs[0], (-1, 168, 168, 3))
        out = self.model.predict(obs)
        action = np.argmax(out[0])
        
        return action #input_map[action]
        
    def remember(self, obs, action, reward, obsn, done):
        self.memory.append([obs[0], action, reward, obsn[0], done])
        if len(self.memory)>256:
            del self.memory[0]
        
    def train(self):
        experience = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))

        obs, actions, rewards, obsn, done = zip(*experience)
        obs = np.asarray(obs)
        
        reward_predicted = self.model.predict(obs)
        Q_sa = self.model.predict(np.asarray(obsn))
        
        for i, ex in enumerate(experience):
            reward_predicted[i][actions[i]] = rewards[i]
            if not done[i]:
                reward_predicted[i][actions[i]] += GAMMA * np.max(Q_sa[i])
                
        self.model.fit(obs, reward_predicted, epochs=10, verbose=0)
        