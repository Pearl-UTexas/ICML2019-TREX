import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import preprocess, normalize_state
import sys
from differentiable_sorting.torch import bitonic_matrices, diff_sort, diff_argsort

def sort(x):
	matrices = bitonic_matrices(len(x))
	sorted_input = diff_sort(matrices, x)
	return sorted_input

def LB_div(gt,r):
	# gt - ground truth ranking
	# r - scores from a reward network

	# sort reward scores r
	sorted_r = sort(r)
	r_permuted = r.index_select(0, gt)
	penalty = torch.tensor(0).float()
	# iterate over trajectories
	for i in range(len(r)):
		# sorted scores, scores permuted acc to GT ranking (still non-differentiable)
		penalty += (sorted_r[i] - r_permuted[i])*i
	try:
		assert(penalty>-1e5)
	except:
		print(r, sorted_r, penalty)
	return max(penalty,torch.tensor(0).float())

def generate_novice_demos(env, env_name, agent, model_dir, num_checkpoints):
	# generate several rollouts for each of the 5 checkpoints
	if env_name == "breakout":
		cpts = [1,125,350,500]#,800]
	elif env_name == "pong":
		cpts = [25,225,350,500]#,800]  #this will hopefully test the bregman divergence since trex will struggle since last demo is shorter than previous two
	elif env_name == "spaceinvaders":
		cpts = [25,125,350,675]#,875]
	else:
		print("no checkpoints specified for ", env_name)
		sys.exit()

	assert(len(cpts)==num_checkpoints)

	checkpoints = []
	for i in cpts:
		if i < 10:
			checkpoints.append('0000' + str(i))
		elif i < 100:
			checkpoints.append('000' + str(i))
		elif i < 1000:
			checkpoints.append('00' + str(i))
		elif i < 10000:
			checkpoints.append('0' + str(i))
	print(checkpoints)

	demonstrations = []
	learning_returns = []
	learning_rewards = []
	for checkpoint in checkpoints:

		model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
		if env_name == "seaquest":
			model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

		agent.load(model_path)
		episode_count = 1
		for i in range(episode_count):
			done = False
			traj = []
			gt_rewards = []
			r = 0

			ob = env.reset()
			steps = 0
			acc_reward = 0
			while True:
				action = agent.act(ob, r, done)
				ob, r, done, _ = env.step(action)
				#env.render()
				if env_name == "pong":
					ob_processed = normalize_state(ob)
				else:
					ob_processed = preprocess(ob, env_name)
				ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
				traj.append(ob_processed)

				gt_rewards.append(r[0])
				steps += 1
				acc_reward += r[0]
				if done:
					print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
					break
			print("traj length", len(traj))
			print("demo length", len(demonstrations))
			demonstrations.append(traj)
			learning_returns.append(acc_reward)
			learning_rewards.append(gt_rewards)

	return demonstrations, learning_returns, learning_rewards



def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, num_checkpoints):
	#collect training data into five separate lists corresponding to checkpoints
	max_traj_length = 0
	training_obs = []
	training_labels = []

	#add full trajs (for use on Enduro)
	for n in range(num_trajs):
		traj_samples = []
		start_indices = []
		 
		# only add trajectories that are different returns
		# pick random trajectories per checkpoint
		for i in range(num_checkpoints):
			traj_samples.append([])
			start_indices.append(np.random.randint(6))

		#create random partial trajs by finding random start frame and random skip frame
		step = np.random.randint(3,7)

		for i in range(num_checkpoints):
			traj_samples[i] = demonstrations[i][start_indices[i]::step]  #slice(start,stop,step)

		# input demonstrations are already sorted
		label = range(num_checkpoints) #[0,1,2,3,4]

		training_obs.append(tuple(traj_samples))
		training_labels.append(label)
		traj_sample_lengths = [len(t) for t in traj_samples]
		max_traj_length = max(max_traj_length, max(traj_sample_lengths))


	#fixed size snippets with progress prior
	for n in range(num_snippets):
		traj_samples = []
		start_indices = [0]*num_checkpoints

		#only add trajectories that are different returns
		#pick random demonstrations from each of the checkpoints
		for i in range(num_checkpoints):
			traj_samples.append([])

		#create random snippets
		#find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
		demo_lengths = [len(d) for d in demonstrations]
		min_length = min(demo_lengths)
		rand_length = np.random.randint(min_snippet_length, max_snippet_length+1)

		last_i, last_j = 0, min_length - rand_length + 1
		for i in range(num_checkpoints):
			#pick better traj snippet to be later than worse traj snippet
			if i >0:
				last_j = demo_lengths[i] - rand_length + 1
			ti_start = np.random.randint(last_i, last_j)
			last_i = ti_start
			
			traj_samples[i] = demonstrations[i][ti_start:ti_start+rand_length:2] #skip every other framestack to reduce size

		traj_sample_lengths = [len(t) for t in traj_samples]
		max_traj_length = max(max_traj_length, max(traj_sample_lengths))
		
		# input demonstrations are already sorted
		label = range(num_checkpoints) #[0,1,2,3,4]

		training_obs.append(tuple(traj_samples))
		training_labels.append(label)

	print("maximum traj length", max_traj_length)
	return training_obs, training_labels




class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
		self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
		self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
		self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 1)



	def cum_return(self, traj):
		'''calculate cumulative return of trajectory'''
		sum_rewards = 0
		sum_abs_rewards = 0
		x = traj.permute(0,3,1,2) #get into NCHW format
		#compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
		x = F.leaky_relu(self.conv1(x))
		x = F.leaky_relu(self.conv2(x))
		x = F.leaky_relu(self.conv3(x))
		x = F.leaky_relu(self.conv4(x))
		x = x.view(-1, 784)
		x = F.leaky_relu(self.fc1(x))
		r = F.relu(self.fc2(x)) #TODO: this is different from original trex so need to change the baselines code too!
		sum_rewards += torch.sum(r)
		sum_abs_rewards += torch.sum(torch.abs(r))
		return sum_rewards, sum_abs_rewards



	def forward(self, trajs):
		'''compute cumulative return for each trajectory and return logits'''
		cum_r, abs_r = [], []
		for i in range(len(trajs)):
			cum_r_i, abs_r_i = self.cum_return(trajs[i])
			cum_r.append(cum_r_i)
			abs_r.append(abs_r_i)
		
		cum_r = [cum_r_i.unsqueeze(0) for cum_r_i in cum_r]
		return torch.cat(tuple(cum_r),0), sum(abs_r)



# Train the network with LB divergence as loss function
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir, num_cpts):
	#check if gpu available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Assume that we are on a CUDA machine, then this should print a CUDA device:
	print(device)
	# loss_criterion = nn.CrossEntropyLoss()

	# use cpu for now
	cum_loss = 0.0
	training_data = list(zip(training_inputs, training_outputs))
	for epoch in range(num_iter):
		np.random.shuffle(training_data)
		training_obs, training_labels = zip(*training_data)
		for i in range(len(training_labels)):
			# get 5 samples from 5 distinct checkpoints 
			trajs = []
			labels = np.array(training_labels[i]) # a ground truth ranking of five trajectories (randomize order)
			labels = torch.from_numpy(labels)

			for k in range(num_cpts):
				traj_k = training_obs[i][k]
				traj_k = np.array(traj_k)
				traj_k = torch.from_numpy(traj_k).float()
				trajs.append(traj_k)
			
			#zero out gradient
			optimizer.zero_grad()

			#forward + backward + optimize
			outputs, abs_rewards = reward_network.forward(trajs)
			# outputs = outputs.unsqueeze(0)
			# outputs = outputs.to(device)

			lb_div = LB_div(labels, outputs) # should be greater than or equal to 0

			loss = lb_div + l1_reg * abs_rewards  # LB divergence as loss
			loss.backward()
			optimizer.step()

			#print stats to see if learning
			item_loss = loss.item()
			cum_loss += item_loss
			if i % 500 ==  499:
				#print(i)
				print("epoch {}:{} loss {} lb_div {} l1_reg {}".format(epoch,i, cum_loss, lb_div, l1_reg*abs_rewards))
				print(abs_rewards)
				print('labels: ', labels, 'outputs: ', outputs)
				cum_loss = 0.0
				print("check pointing")
				torch.save(reward_net.state_dict(), checkpoint_dir)
	print("finished training")



def calc_accuracy(reward_network, training_inputs, training_outputs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# loss_criterion = nn.CrossEntropyLoss()
	num_correct = 0.
	with torch.no_grad():
		for i in range(len(training_inputs)):
			label = training_outputs[i]
			trajs = []

			for k in range(len(training_inputs[i])):
				traj_k = training_inputs[i][k]
				traj_k = np.array(traj_k)
				traj_k = torch.from_numpy(traj_k).float().to(device)
				trajs.append(traj_k)

			#forward to get logits
			outputs, abs_return = reward_network.forward(trajs)
			_, pred_label = torch.max(outputs,0)
			if pred_label.item() == label:
				num_correct += 1.
	return num_correct / len(training_inputs)



def predict_reward_sequence(net, traj):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	rewards_from_obs = []
	with torch.no_grad():
		for s in traj:
			r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
			rewards_from_obs.append(r)
	return rewards_from_obs

def predict_traj_return(net, traj):
	return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
	parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
	parser.add_argument('--seed', default=0, help="random seed for experiments")
	parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
	parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
	parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")
	parser.add_argument('--num_epochs', default = 1, type = int, help = "number of times to run through training data")
	parser.add_argument('--num_checkpoints', default = 4, type = int, help = "number of checkpoints to sample set of ranked trajectories")

	args = parser.parse_args()
	env_name = args.env_name
	if env_name == "spaceinvaders":
		env_id = "SpaceInvadersNoFrameskip-v4"
	elif env_name == "mspacman":
		env_id = "MsPacmanNoFrameskip-v4"
	elif env_name == "videopinball":
		env_id = "VideoPinballNoFrameskip-v4"
	elif env_name == "beamrider":
		env_id = "BeamRiderNoFrameskip-v4"
	else:
		env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

	env_type = "atari"
	print(env_type)
	#set seeds
	seed = int(args.seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)

	print("Training reward for", env_id)
	num_trajs =  args.num_trajs
	num_snippets = args.num_snippets
	num_checkpoints = args.num_checkpoints
	min_snippet_length = 50 #min length of trajectory for training comparison
	maximum_snippet_length = 50

	lr = 0.00005
	weight_decay = 0.001
	num_iter = args.num_epochs  #use at least 4 for pong. 1 for others
	l1_reg=0.0
	stochastic = True

	env = make_vec_env(env_id, 'atari', 1, seed,
					   wrapper_kwargs={
						   'clip_rewards':False,
						   'episode_life':False,
					   })


	env = VecFrameStack(env, 4)
	agent = PPO2Agent(env, env_type, stochastic)

	demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir, num_checkpoints)

	#sort the demonstrations according to ground truth reward to simulate ranked demos

	demo_lengths = [len(d) for d in demonstrations]
	print("demo lengths", demo_lengths)
	max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
	print("max snippet length", max_snippet_length)

	print(len(learning_returns))
	print(len(demonstrations))
	print([a[0] for a in zip(learning_returns, demonstrations)])
	demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

	sorted_returns = sorted(learning_returns)
	print(sorted_returns)

	training_obs, training_labels = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, num_checkpoints)
	print("num training_obs", len(training_obs))
	print("num_labels", len(training_labels))

	# Now we create a reward network and optimize it using the training data.
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	reward_net = Net()
	# reward_net.to(device)
	import torch.optim as optim
	optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
	learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path, num_checkpoints)
	#save reward network
	torch.save(reward_net.state_dict(), args.reward_model_path)

	#print out predicted cumulative returns and actual returns
	with torch.no_grad():
		pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
	for i, p in enumerate(pred_returns):
		print(i,p,sorted_returns[i])

	print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))
