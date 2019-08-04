import gym
import retro
#import retro.Super_mario_brothers.nes
import time
import numpy as np
from PIL import Image
import skimage
import itertools
print(retro.data.list_games())
#gym_pull.pull('github.com/ppaquette/gym-super-mario')
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
#import gym_super_mario_bros
#from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
#env = gym_super_mario_bros.make('SuperMarioBros-8-4-v0')
#env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
#env = gym.make('FreewayDeterministic-v4')
#gym.undo_logger_setup()
env = retro.make('SuperMarioBros-Nes', 'Level1-1')
buttons = env.BUTTONS
env = LimitedDiscreteActions(env, buttons)
print('env id', env.spec.id)
exit()

#action_space = env.action_space
#print("action space", env.action_space)
observation = env.reset()
print("obs shape", observation.shape)
print(env.unwrapped.get_action_meanings())
#env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

done = False
t = 0

start = time.time()
num_lives = env.unwrapped.ale.lives()
for _ in range(10000):
	action = env.action_space.sample()
	#print("action:", action)
	#print(env.action_space)
	obs, reward, done, info = env.step(action) # take a random action
	#print('reward', reward)
	img = obs 
	img = np.array(Image.fromarray(img).resize([84,110]))[110-84:,0:84,:]

	# lives = env.unwrapped.ale.lives()
	# if lives < num_lives:
	# 	done = True
	# 	print('death')
	# num_lives = lives 
	#scipy.misc.imshow(img)
    #img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    #img= img/256.00
    #img = img.reshape([1,self.image_size,self.image_size,1])
	#if _ > 0 and _ % 100 == 0:
		#scipy.misc.imshow(img)

	if done :
		env.reset()
	#print(action)
	env.render()

time_taken = time.time() - start
fps = 10000 / time_taken
print('Pil time', time_taken, 'PIL fps', fps)




env.close()




class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask
