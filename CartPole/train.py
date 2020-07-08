import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from tqdm import tqdm
from itertools import count
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from replay_memory import *
from model import *


def get_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

def clean_frame(frame):
	if frame.all()  == None:
		return np.zeros(1, 80*80)
	else:
		# set background to black
		frame[frame == 144] = 0
		frame[frame == 109] = 0
		
		# greyscale
		frame[frame != 0] = 255
		return frame
	
def frame_to_tensor(frame):
	return torch.from_numpy(frame.astype(np.float32).ravel()).unsqueeze(0)

def preprocess(frame1, frame2):
	return frame_to_tensor(clean_frame(frame2)) - frame_to_tensor(clean_frame(frame1))

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
	# crashes if k/l are zero
	if k <= 0 or l <= 0: 
		if not err: 
			print(f"Number of repeats must be larger than 0, k: {k}, l: {l}, returning default array!")
			err.append('logged')
			return rgb_array

	# repeat the pixels k times along the y axis and l times along the x axis
	# if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
	return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_durations(save_dir=None):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    if save_dir != None:
        plt.imsave(save_dir)




def train(num_episodes=100):
    
    env.reset()
    
        
    steps_done = 0

    episode_durations = []	
    
    for i_episode in tqdm(range(num_episodes)):
        
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        
        for t in count():
            # Select and perform an action
            action = select_action(state, steps_done)
            steps_done+=1
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
            break

            # Update the target network, copying all weights and biases in DQN
        
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


    env.render()
    env.close()
    plt.show()


if __name__=='__main__':
    env = gym.make('CartPole-v0').unwrapped
    
    # initialize screen
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape 


    n_actions = env.action_space.n
    memory = ReplayMemory(10000)

    resize  = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    train()

