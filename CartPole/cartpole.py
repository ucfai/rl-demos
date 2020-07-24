import sys
import gym
import time
import numpy as np
import os
from gym.envs.classic_control import rendering
from tqdm import tqdm
from collections import namedtuple
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 80*80
hidden_dim = 64
output_dim = 1

MODEL_DIR = os.getcwd() + '/models/'

# keep track of observed states
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# buffer that holds recently seen states
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
        def __init__(self, h, w, outputs):
                super(DQN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
                self.bn3 = nn.BatchNorm2d(32)    
        
                # calculate output dimensions of convolutional layer
                def conv2d_size_out(size, kernel_size = 5, stride = 2):
                    return (size - (kernel_size - 1) - 1) // stride  + 1
                

                convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
                convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
                linear_input_size = convw * convh * 32
                self.head = nn.Linear(linear_input_size, outputs)

        # feed input data through the NN
        def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                return self.head(x.view(x.size(0), -1))
        

resize  = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
        


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
        # repeat kinda crashes if k/l are zero
        if k <= 0 or l <= 0: 
                if not err: 
                        print(f"Number of repeats must be larger than 0, k: {k}, l: {l}, returning default array!")
                        err.append('logged')
                        return rgb_array

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

 

def main():
        
        viewer = rendering.SimpleImageViewer()
        env = gym.make('Pong-v0')
        
        input_dim = 80*80
        hidden_dim = 32
        output_dim = 1

        model = Model()
        
        loss = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)            

        epochs = 5
        
        for e in tqdm(range(epochs)): 
                env.reset()
                action=0
                play = True
                
                while play:
                        rgb = env.render('rgb_array')
                        action = 2 # 2 for up, 3 for down
                        env.step(3)[0]
                        #upscaled=repeat_upsample(rgb,5, 5)
                        cleaned = clean_frame(rgb)
                        viewer.imshow(rgb)
                        observation, reward, done, info = env.step(action)
                                  
                        if done:
                                play = False
                                
                                # fit model
                                '''     
                                optimizer.zero_grad()
                                # forward + backward + optimize
                                outputs = net(inputs)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                '''

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_durations():
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

 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__=='__main__':
        env = gym.make('CartPole-v0').unwrapped
        env.reset()
        
        BATCH_SIZE = 2e20
        GAMMA = 0.999
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        TARGET_UPDATE = 10

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        init_screen = get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        n_actions = env.action_space.n

        policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(10000)

        steps_done = 0
        
        if len(sys.argv) > 1:
            num_episodes = int(sys.argv[1])
        else:
            num_episodes = 50

        episode_durations = []  

        for i_episode in tqdm(range(num_episodes)):
                # Initialize the environment and state
                env.reset()
                last_screen = get_screen()
                current_screen = get_screen()
                state = current_screen - last_screen
                for t in count():
                        # Select and perform an action
                        action = select_action(state)
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

        
        torch.save(policy_net.state_dict(), MODEL_DIR + str(num_episodes) + '-epochs.model')
        print('Complete')
        env.render()
        env.close()
