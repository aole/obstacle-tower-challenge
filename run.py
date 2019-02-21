from obstacle_tower_env import ObstacleTowerEnv
import sys, os, re
import argparse
from mota import Model
from tqdm import trange
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim

input_map = []
for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(3):
                input_map.append([i, j, k, l])
                
model = Model()
MAX_EPISODES = 20

def run_test(env, episode_number):
    print('Running test!')
    total_reward = 0.0
    obs = env.reset()
    
    imdata = obs[0]*255
    im = [Image.fromarray(imdata.astype(np.uint8))]
    
    for i in range(5000): #trange(5000):
        #
        action = model.act(obs, False)
        obsn, reward, done, _ = env.step(input_map[action])
        
        imdata = obsn[0]*255
        im.append(Image.fromarray(imdata.astype(np.uint8)))
    
        obs = obsn
        total_reward += reward
        
        if done:
            break
            
    im[0].save(f'test{episode_number}.gif', save_all=True, append_images=im[1:], duration=200, loop=0)

    return total_reward

def run_episode(env, episode_number, save=True):
    total_reward = 0.0
    obs = env.reset()
    
    if save:
        imdata = obs[0]*255
        im = [Image.fromarray(imdata.astype(np.uint8))]
    
    #for i in range(5000): 
    for i in trange(5000):
        #
        action = model.act(obs)
        obsn, reward, done, _ = env.step(input_map[action])
        
        if save:
            imdata = obsn[0]*255
            im.append(Image.fromarray(imdata.astype(np.uint8)))
    
        # increase reward impact
        reward *= 10
        #
        #if done and reward==0:
        #    reward = -10
        
        # if image same as previous
        # staring/standing still?
        #if len(model.memory)>2 and (np.array_equal(model.memory[-1][0], obsn[0]) or np.array_equal(model.memory[-2][0], obsn[0])):
        if len(model.memory)>2 and (compare_ssim(model.memory[-1][0], obsn[0], multichannel=True)>0.9 or compare_ssim(model.memory[-2][0], obsn[0], multichannel=True)>0.9):
            reward -= 1
            
        if reward!=0:
            model.train(obs, action, reward, obsn, done)
            
        #if not done: # reward for surviving
            #reward += 0.1
        model.remember(obs, action, reward, obsn, done)
        
        obs = obsn
        total_reward += reward
        
        if done:
            break
            
    if save:
        im[0].save(f'episode{episode_number}.gif', save_all=True, append_images=im[1:], duration=200, loop=0)

    model.train_batch()
    
    return total_reward

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env, 0)
        env.reset()

if __name__ == '__main__':
    # delete files generated in previous run
    dir = '.'
    files = os.listdir(dir)
    for file in files:
        if re.search('(episode|test)\d+\.gif', file):
            os.remove(os.path.join(dir, file))
            
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    print('opening env...')
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, retro=False, realtime_mode=False)
    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        episode_reward = 0
        episode_number = 0
        while True:
            episode_number += 1
            print(f"Episode {episode_number}...")
            episode_reward = run_episode(env, episode_number, episode_number%5==0 or episode_number==1)
            print(f"...reward: {episode_reward}. Epsilon: {model.epsilon}")
            
            if episode_number%10==0:
                episode_reward = run_test(env, episode_number)
                print(f'...test reward:{episode_reward}.')
                
            if episode_number>=MAX_EPISODES:
                break

    env.close()

