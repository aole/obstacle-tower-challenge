from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
from mota import Model

input_map = []
for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(3):
                input_map.append([i, j, k, l])
                
model = Model()

def run_episode(env):
    done = False
    total_reward = 0.0
    obs = env.reset()
    
    while not done:
        #
        action = model.act(env, obs)
        obsn, reward, done, _ = env.step(input_map[action])
        
        if reward:
            print(f'\tReward: {reward}')
            
        model.remember(obs, action, reward, obsn, done)
        model.train()
        
        obs = obsn
        total_reward += reward
        
    return total_reward

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
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
            episode_reward = run_episode(env)
            print(f"...reward: {episode_reward}")
            
            if episode_number>20:
                break

    env.close()

