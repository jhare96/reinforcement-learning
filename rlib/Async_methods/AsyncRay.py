import ray
import ray.experimental.tf_utils
import tensorflow as tf
import numpy as np
import gym
import os, time
import threading
import multiprocessing as mp
from A2C import ActorCritic
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
import copy

ray.init()

@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self):
        return self.weights


@ray.remote
class RayAsyncWorker(object):
    def __init__(self, worker_id, server, env_constructor, model_constructor, nsteps, train_log_dir=None, env_args={}, model_args={}):
        config = tf.ConfigProto() #GPU 
        config.gpu_options.allow_growth=True #GPU
        config.log_device_placement = True
        sess = tf.Session(config=config)
        #config = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
        self.sess = tf.Session(config=config)
        self.worker_id = worker_id
        self.server = server
        self.tmax = nsteps
        
        #with tf.device('/device:cpu:0'):
        self.model = model_constructor(**model_args)
        
        self.env = env_constructor(**env_args)
        self.weights = ray.experimental.tf_utils.TensorFlowVariables(self.model.loss, self.sess)
        # self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # self.placeholders = [tf.placeholder(dtype=w.dtype, shape=w.get_shape().as_list()) for w in self.weights]
        

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.state = self.env.reset()
        


    def update(self,):
        weights_np = ray.get(self.server.pull.remote())
        self.weights.set_weights(weights_np)
        # weights_tf = [tf.convert_to_tensor(weight) for weight in weights_np]
        # update_var = [old_weight.assign(new_weight) for (old_weight, new_weight) in 
        #     zip(self.weights, weights_tf)]
        # feed_dict = {}
        # for i in range(len(weights_np)):
        #     feed_dict[self.placeholders[i]] = weights_np[i]
        # self.sess.run(self.placeholders, feed_dict)
        # weights_tf = self.placeholders
        # update_var = [tf.assign(new, old) for (new, old) in 
        #     zip(self.weights, weights_tf)]
        # self.sess.run(self.init)
        # self.sess.run(update_var)
        #return self.sess.run(self.weights)
    
    def trajectory(self):
        start = time.time()
        traj = []
        #self.update()
        for t in range(self.tmax):
            policy, value = self.model.forward(self.sess,self.state.reshape(1,*self.state.shape))
            action = np.argmax(policy)
            next_state, reward, done, info = self.env.step(action)
            traj.append([self.state, action, reward])

            if done:
                self.states = self.env.reset()
            
            #self.server.inc.remote()

        fps = self.tmax / (time.time() - start) 
        return traj, self.worker_id
    






class MPAsyncWorker(mp.Process):
    def __init__(self, worker_id):
        mp.Process.__init__(self)
        self.daemon = True
        self.sess = tf.Session()
        self.env = AtariEnv(gym.make('SpaceInvaders-v0'))
        with tf.variable_scope('worker_' + str(worker_id)):
            self.var = tf.Variable([3.0, 4.0], trainable=True)
        
        self.update_var = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='worker_' + str(worker_id) ), tf.trainable_variables('master'))]

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def update(self):
        self.sess.run(self.update_var)
        return self.sess.run(self.var)
    
    def run(self):
        self.train()
    
    def train(self,):
        rewards = []
        self.env.reset()
        for t in range(100):
            #action = int(self.sess.run(self.var)[0])
            
            action = self.env.action_space.sample()
            print('action', action)
            obs,r, done, info  = self.env.step(action)
            rewards.append(r)
        
        return rewards
    
    



def actor_constructor(model, **model_args):
    return ActorCritic(model, **model_args)

def env_constructor(env_constr, env_id, **env_args):
    return env_constr(gym.make(env_id), **env_args)

def pass_(x):
    pass

def test_graph():
    #mp.set_start_method('spawn') 
    env_id = 'SpaceInvaders-v0'
    train_log_dir = 'logs/AyncWorker/' + env_id + '/'
    env_constr = env_constructor
    model_constr = actor_constructor
    model_args = {'model':Nature_CNN, 'input_size':[84,84,4], 'action_size':6,  'lr':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(32*5),
                    'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    #config = tf.ConfigProto() #GPU 
    #config.gpu_options.allow_growth=True #GPU
    #sess = tf.Session(config=config)
    sess = tf.Session()
    
    
    
    learner = model_constr(**model_args)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    #learner = AsyncWorker(-1, 'Learner', env_constr, model_constr, train_log_dir, {'env_constr':DummyEnv,'env_id':env_id}, model_args)
    
    #workers = [MPAsyncWorker(i) for i in range(2)]
    # tvs = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    tvs = ray.experimental.tf_utils.TensorFlowVariables(learner.loss, sess).get_weights()
    keys, values = list(tvs.keys()), list(tvs.values())
    
    server = ParameterServer.remote(keys, values)
    workers = [RayAsyncWorker.remote(i, server, env_constr, model_constr, 20, train_log_dir, {'env_constr':AtariEnv, 'env_id':env_id}, model_args) for i in range(16)]

    

    #trains = [w.train.remote() for w in workers]
    
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    

    start = time.time()

    # for w in workers:
    #     w.start()
    
    # for w in workers:
    #     w.join()
    #print('results', res)
    
    res = [w.update.remote() for w in workers]
    #res = ray.get(res)
    #print('results', res)
    print('update weights time', time.time()-start)


    trajectories = []
    for worker in workers:
        trajectories.append(worker.trajectory.remote() )
        time.sleep(2)

    i = 0
    start = time.time()
    while True:
        i += 1
        
        rollout, trajectories = ray.wait(trajectories)
        #print('ray wait', rollout)
        rollout, worker_id = ray.get(rollout)[0]
        #print('rolloout', len(rollout[0]), rollout[0][0].shape)
        
        trajectories.append(workers[worker_id].trajectory.remote())
        
        if i % 100 == 0:
            fps = (100*20) / (time.time()-start)
            print('fps', fps)
            start = time.time()
    
    # start = time.time()
    # for t in range(1,1000):
    #     trajectories = [w.trajectory.remote() for w in workers]
    #     rollouts = ray.get(trajectories)

    #     if t % 100 == 0:
    #         time_taken = time.time() - start
    #         fps = 100 * 20 * len(workers) / time_taken
    #         print('fps', fps)
    #         start = time.time()

        

    #res = [w.train() for w in workers]
    #ray.get(res)
    #print('results', res)

    


    

if __name__ == "__main__":
    test_graph()