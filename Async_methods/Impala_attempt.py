import tensorflow as tf
import numpy as np
import threading, multiprocessing
import queue
import gym
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from A2C import ActorCritic
from VecEnv import*
from networks import*


class Worker(threading.Thread):
    def __init__(self, ID, sess, T, device, queue, lock, env_constr, model_constr, daemon, env_args={}, model_args={}, nsteps=200, max_rollouts=5e6):
        threading.Thread.__init__(self, daemon=daemon)
        self.sess = sess
        self.queue = queue
        self.lock = lock
        self.nsteps = nsteps
        self.max_rollouts = int(max_rollouts)
        self.env = Env(env_constr(**env_args))
        if 'gpu' in device.lower():
            with tf.variable_scope('worker_' + str(ID)):
                self.model = model_constr(**model_args)
        else:
            with tf.device(device):
                with tf.variable_scope('worker_' + str(ID)):
                    self.model = model_constr(**model_args)
        
        self.workerID = ID
        self.T = T
        self.update_local = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='worker_' + str(self.workerID) ), tf.trainable_variables('Learner'))]
    
    def run(self):
        self.rollout()
    
    def rollout(self):
        env = self.env
        state = env.reset()
        for roll in range(self.max_rollouts):
            self.sess.run(self.update_local)
            #states = np.zeros((self.nsteps,))
            trajectory = []
            for t in range(self.nsteps):
                policy, value = self.model.forward(self.sess, state[np.newaxis])
                action = np.argmax(policy)
                next_state, reward, done , info = env.step(action)
                trajectory.append([state,action,reward])
                
                with self.lock:
                    self.T[:] += 1
                    #print('T', self.T)
                state = next_state
                
            
            self.queue.put(trajectory)





def actor_constructor(model, **model_args):
    return ActorCritic(model, **model_args)

def env_constructor(env_constr, env_id, **env_args):
    return env_constr(gym.make(env_id), **env_args)

def fold_batch(x):
    rows, cols = x.shape[0], x.shape[1]
    y = x.reshape(rows*cols,*x.shape[2:])
    return y

def main():
    env_id = 'SpaceInvaders-v0'
    train_log_dir = 'logs/Impala/' + env_id + '/'
    env_constr = env_constructor
    model_constr = actor_constructor
    model_args = {'model':Nature_CNN, 'input_size':[84,84,4], 'action_size':6,  'lr':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(32*5),
                    'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    env_args = {'env_constr':AtariEnv, 'env_id':env_id}

    num_workers = 32

    config = tf.ConfigProto(log_device_placement=False) #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.Session(config=config)
    #sess = tf.Session()

    with tf.variable_scope('Learner'):
        learner = model_constr(**model_args)
    
    init = tf.global_variables_initializer()
    sess.run(init)


    T = np.array([0], dtype=np.int32)
    q = multiprocessing.Queue(maxsize=10)
    lock = threading.Lock()
    gpu_workers = ['/device:GPU:0' for i in range(16)]
    cpu_workers = ['/device:cpu:0' for i in range(num_workers-len(gpu_workers))]
    devices = gpu_workers + cpu_workers
    workers = [Worker(i, sess, T, devices[i], q, lock, env_constr, model_constr, True, env_args, model_args) for i in range(num_workers)]

    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    try:
        for w in workers:
            w.start()
        
        t = workers[0].T.copy()
        start = time.time()
        j = 0
        while True:
            #time.sleep(0.1)
            batches = []
            batches = q.get()
            if not q.empty():
                j += 1
                while not q.empty():
                    batches.append(q.get())
            #     states = np.stack([batch[0] for batch in batches])
            #     actions = np.stack([batch[1] for batch in batches])
            #     rewards = np.stack([batch[2] for batch in batches])
            #     print('batches length,', len(batches))
            #     print('sattes', states.shape)
            #     #l = learner.backprop(sess, states, rewards, actions)
                if j % 10 == 0:
                    end = time.time()
                    fps = (workers[0].T - t) / (end-start)
                    print('total steps %i, loss %f, fps %f' %(workers[0].T, 0, fps))
                    t = workers[0].T.copy()
                    start = time.time()
        
    except KeyboardInterrupt:
        pass
    
    finally:
        for w in workers:
            w.env.close()
            w.join()


if __name__ == "__main__":
    main()