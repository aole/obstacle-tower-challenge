from obstacle_tower_env import ObstacleTowerEnv
import sys, os, re, time, random
import argparse
from mota import Model, ACTION_MAP, Actor, Critic
from tqdm import trange
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import logging
from skimage.transform import resize

logger = logging.getLogger("gym_unity")

MAX_EPISODES = 10000
EPISODE_DONE_PENALTY = False
COMPARE_PREVIOUS_FRAMES = False
TRAIN_REWARD = False
STEP_REWARD = 0
TRAIN_BATCH_EVERY_STEP = False
ACTION_REWARD_HACKS = (0,0,0,0,0,0,0)
TRAIN_EVERY = 1

model = Model()
actor = Actor()
critic = Critic()

fastest_to_floor = [10000, 10000, 10000, 10000, 10000, 10000, 10000]

def run_episode(env, episode_number, test=False):
    starting_floor = random.choice([0,1])
    if test:
        print(f'TST: {episode_number}-{starting_floor}...')
    else:
        print(f'EPS: {episode_number}-{starting_floor}...')
        
    total_reward = 0.0
    
    env.floor(starting_floor)
    num_floors = starting_floor
    
    obs = env.reset()
    
    obs_im = resize(obs[0], (84,84,3), anti_aliasing=False, mode='constant')
    
    imdata = obs[0]*255
    im = [Image.fromarray(imdata.astype(np.uint8))]
    
    for frames in range(50000):
    #for i in trange(5000):
        #
        action = model.act(obs, test==False)
        repeat_action = 1
        if action in [3,4]:
            repeat_action = 5
        
        for ra in range(repeat_action):
            obsn, reward, done, _ = env.step(ACTION_MAP[action])
            
        obsn_im = resize(obs[0], (84,84,3), anti_aliasing=False)
        if reward==1:
            num_floors += 1
            next_floor_found = True
            if frames<fastest_to_floor[num_floors]:
                fastest_to_floor[num_floors] = frames
            reward = 10
                
        imdata = obsn[0]*255
        im.append(Image.fromarray(imdata.astype(np.uint8)))
    
        if not test:
            
            # reward for taking certain actions
            reward += ACTION_REWARD_HACKS[action]
            
            # if image same as previous
            # staring/standing still?
            
            if COMPARE_PREVIOUS_FRAMES:
                if len(model.memory)>2 and (compare_ssim(model.memory[-1][0], obsn_im, multichannel=True)>0.95 or compare_ssim(model.memory[-2][0], obsn_im, multichannel=True)>0.95):
                    reward -= 1
            
            # reward for every step
            reward += STEP_REWARD
            
            if done and not num_floors>starting_floor:
                if EPISODE_DONE_PENALTY:
                    reward -= 1
                
            model.remember(obs_im, action, reward, obsn_im, done)
            
            if round(reward)!=0:
                print('\t'+'* '*num_floors +f'Reward: {reward} ({frames})', end='')
                if TRAIN_REWARD:
                    print(' Training...', end='')
                    st = time.time()
                    model.train_batch()
                    print(f'{round(time.time()-st, 2)}!')
                else:
                    print()
            if TRAIN_BATCH_EVERY_STEP:
                print(f'\tBatch: Floors:{num_floors}.')
                model.train_batch()
                
            obs = obsn
        
        total_reward += reward
        
        if done:
            break
            
    if test:
        im[0].save(f'test{episode_number}_{starting_floor}-{num_floors}.gif', save_all=True, append_images=im[1:], duration=200, loop=0)
    elif num_floors>starting_floor:
        im[0].save(f'episode{episode_number}_{starting_floor}-{num_floors}.gif', save_all=True, append_images=im[1:], duration=200, loop=0)

    if not TRAIN_BATCH_EVERY_STEP and not test:
        print(f'\tLvl: +{num_floors-starting_floor} ({round(total_reward,3)}). Eps:{round(model.epsilon,3)}...', end='')
        if episode_number % TRAIN_EVERY==0:
            st = time.time()
            loss = model.train_batch()
            '''
            if loss>10:
                model.epsilon = max(0.1, model.epsilon-0.1)
            '''
            if loss<0.01:
                model.epsilon = min(0.9, model.epsilon+0.1)
            print(f'L:{round(loss,5)} T:{round(time.time()-st, 2)}')
        else:
            print('.')
    elif test:
        print(f'\tLvl: +{num_floors-starting_floor} ({frames}).')
        
    return frames
    
def run_evaluation(env):
    while not env.done_grading():
        run_episode(env, 0)
        env.reset()

if __name__ == '__main__':
    # delete files generated in previous run
    dir = '.'
    files = os.listdir(dir)
    for file in files:
        if re.search('^(episode|test|frame).+\.(gif|jpg)$', file):
            os.remove(os.path.join(dir, file))
            
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, retro=False, realtime_mode=False)
    logger.setLevel(logging.WARNING)
    env.seed(4)
    
    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        total_frames = 0
        episode_number = 0
        while True:
            episode_number += 1
            total_frames += run_episode(env, episode_number)
            
            if episode_number%200==0:
                print(f'Total Frames: {total_frames}')
                episode_reward = run_episode(env, episode_number, test=True)
                
            if episode_number>=MAX_EPISODES:
                break

    env.close()

