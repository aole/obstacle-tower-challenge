# Learning Obstacle Tower Agent (MOTA)

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

np.random.seed(4)
tf.set_random_seed(4)
tf.logging.set_verbosity(tf.logging.INFO)

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
IMAGE_SHAPE = (84,84,3)
TF_IMAGE_SHAPE = (1,84,84,3)

GAMMA       = 0.99
BATCH_SIZE  = 64
BATCH_EPOCH = 1
BATCH_LOOP = 10
MEMORY_SIZE = 2024

USE_ADVANTAGE = False
SPREAD_REWARD = False
SPREAD_REWARD_DECAY = 0.9
LEARNING_RATE = 0.01
USE_ACTION_PROBABILITIES = True

ACTION_HACKS = True
MOVE_FORWARD_PROB = 2
CAMERA_PROB = .9
JUMP_PROB = .7

DENSE_LAYER_SIZE = 512

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm
    
def softmax(v):
    x = v[0]
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    
class Actor:
    def __init__(self):
        self.s = tf.placeholder(tf.float32, TF_IMAGE_SHAPE, "state")
    
class Critic:
    def __init__(self):
        pass
    
class Model:
    def __init__(self):
        # probabilities
        self.prob = [1]*ACTION_SIZE
        if ACTION_HACKS:
            self.prob[1] = MOVE_FORWARD_PROB
            self.prob[5] = self.prob[6] = CAMERA_PROB
            self.prob[3] = self.prob[4] = CAMERA_PROB
            
        self.prob = normalize(self.prob)
        #
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        
        self.memory = []
        
        self.model = Sequential()
    
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SHAPE))
        #self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        #self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
            
        self.model.add(Flatten())
        self.model.add(Dense(DENSE_LAYER_SIZE, activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(ACTION_SIZE, activation='linear'))

        self.model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE), loss='mse')
        
    def act(self, obs, explore=True):
        if explore and self.epsilon > self.epsilon_min and np.random.random()<self.epsilon:
            self.epsilon *= self.epsilon_decay
            return np.random.choice(ACTION_SIZE, 1, p=self.prob)[0]
            
        obs = np.reshape(obs[0], (-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
        out = self.model.predict(obs)
        p = softmax(out)
        
        if USE_ACTION_PROBABILITIES:
            action = np.random.choice(ACTION_SIZE, 1, p=p)[0]
        else:
            action = np.argmax(out[0])
        
        return action #input_map[action]
        
    def remember(self, obs, action, reward, obsn, done):
        self.memory.append([obs, action, reward, obsn, done])
        if len(self.memory)>MEMORY_SIZE:
            found = False
            # delete memories that are not relevant
            for i in range(len(self.memory)):
                if abs(self.memory[i][2]) < 1:
                    del self.memory[i]
                    found = True
                    break
            # if all are relevant, delete the oldest
            if not found:
                del self.memory[0]
        
        if SPREAD_REWARD and reward!=0 and len(self.memory)>2:
            r = reward
            for i in range(-2, -len(self.memory), -1):
                r *= SPREAD_REWARD_DECAY
                if r<=0.01:
                    break
                self.memory[i][2] += r
                
    def train(self, obs, action, reward, obsn, done):
        obs = np.reshape(obs[0], (-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
        obsn = np.reshape(obsn[0], (-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
        
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
        h = []
        for j in range(BATCH_LOOP):
            experience = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))

            obs, actions, rewards, obsn, done = zip(*experience)
            obs = np.asarray(obs)
            
            reward_predicted = self.model.predict(obs)
            Q_sa = self.model.predict(np.asarray(obsn))
            
            for i, ex in enumerate(experience):
                reward_predicted[i][actions[i]] = rewards[i]
                if not done[i]:
                    reward_predicted[i][actions[i]] += GAMMA * np.max(Q_sa[i])
                    
            hist = self.model.fit(obs, reward_predicted, epochs=BATCH_EPOCH, verbose=0)
            h.append(hist.history['loss'])
        
        return np.mean(h)
                
if __name__ == '__main__':
    model = Model()
    
    for i, l in enumerate(model.model.layers):
        w = l.get_weights()
        if type(w)==list:
            for j,k in enumerate(w):
                print(i, j, k.shape, np.max(k))
        else:
            print(i, w.shape, np.max(w))
            
    for t in range(100):
        for i in range(10):
            obs = np.random.random((1,IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
            model.act(obs)
            obsn = np.random.random((IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
            a = np.random.choice(ACTION_SIZE, 1)[0]
            r = np.random.choice(10, 1)[0]
            model.remember(obs[0], a, r, obsn, False)
        print('training...')
        model.train_batch()
        
        for i, l in enumerate(model.model.layers):
            w = l.get_weights()
            if type(w)==list:
                for j,k in enumerate(w):
                    print(i, j, k.shape, np.max(k))
            else:
                print(i, w.shape, np.max(w))
            
    print('done!')