import tensorflow as tf 
import gym 
import numpy as np 
import multiprocessing, threading
import time, copy
import queue
import sys
import argparse


from A2C import ActorCritic
from rlib.utils.VecEnv import*
from rlib.networks.networks import*


class Actor(ActorCritic):
    def __init__(self, model, action_size, lr=0.001, lr_final=1e-06, decay_steps=600000.0, grad_clip=0.5, **model_args):
        ActorCritic.__init__(self, model, action_size, lr=lr, lr_final=lr_final, decay_steps=decay_steps, grad_clip=grad_clip, **model_args)
        self.sess = None

    def forward(self, state):
        sess = self.sess
        #print('sess', sess)
        return sess.run([self.policy_distrib, self.V], feed_dict={self.state:state})
    
    def backprop(self, state, y, a):
        sess = self.sess
        return super().backprop(sess, state, y, a)


FLAGS = None
NSTEPS = 20

class Runner(threading.Thread):
    def __init__(self, env, model, daemon):
        threading.Thread.__init__(self, daemon=daemon)
        self.env = env 
        self.model = model
        self.queue = queue.Queue(10)
        self.start_time = time.time()
    
    def run(self):
        self.run_()
    
    def run_(self,):
        state = self.env.reset()
        while True:
            rollout = []
            for t in range(NSTEPS):
                policy, value = self.model.forward(state[np.newaxis])
                action = np.argmax(policy)
                next_state, reward, done, info = self.env.step(action)
                rollout.append((state,action,reward,done))
                state = next_state
                #print('t', t)

            self.queue.put(rollout)
    
    def process(self,sess):

        rollout = self.queue.get()
        #print('rollout', len(rollout))
        while not self.queue.empty():
            try:
                rollout.append(self.queue.get_nowait())
            except queue.Empty:
                break

        states = np.stack([batch[0] for batch in rollout])
        actions = np.stack([batch[1] for batch in rollout])
        rewards = np.stack([batch[2] for batch in rollout])
        dones = np.stack([batch[3] for batch in rollout])
        #states, actions, rewards, dones = rollout
        states, actions, rewards, dones = np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(dones)
        _, l = sess.run([self.model.optimiser, self.model.loss], feed_dict={self.model.state : states, self.model.y : rewards, self.model.actions: actions})

        time_taken = time.time() - self.start_time
        fps = len(rollout) / time_taken
        print('fps %f' %(fps))
        self.start_time = time.time()


    



def episode(i, sess, model, env):
    episode_reward = []
    state = env.reset()
    start = time.time()
    t=0
    while True:
        #print('done', done)
        policy, value = model.forward(sess, state[np.newaxis])
        action = np.argmax(policy)
        next_state, reward, done, info = env.step(action)
        episode_reward.append(reward)
        model.backprop(sess, state[np.newaxis], np.asarray([reward]), np.asarray([action]))
        state = next_state
        
        t+= 1

        if done:
            print('t', t)
            time_taken = time.time() - start
            fps = t /time_taken
            print('episode %i, reward %f, time_taken %f, fps %f' %(i,np.sum(episode_reward),time_taken, fps))
            break

def main(ps_hosts, worker_hosts, job_name, task_index):
    #ps_hosts = FLAGS.ps_hosts.split(",")
    #worker_hosts = FLAGS.worker_hosts.split(",")

    if job_name == 'worker':
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
    else:

    #config.log_device_placement=True
    #config.gpu_options.allow_growth=True #GPU
        config = tf.ConfigProto()

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                            job_name=job_name,
                            task_index=task_index,
                            config=config)
    
    print('target', server.target)
    
    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d/cpu:0" % task_index,
            cluster=cluster)):
            ac_cnn_args = {'input_size':[84,84,4], 'action_size':6,  'lr':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(32*5), 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
            model = Actor(Nature_CNN, **ac_cnn_args)
            loss = model.loss
            global_step = tf.contrib.framework.get_or_create_global_step()
            env = AtariEnv(gym.make('SpaceInvaders-v0'))
            
        runner = Runner(env, model, False)
        #runner.join()
            #train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            #train_op = tf.Variable([1,2,3,4,5])


            # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=10)]
           
            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
        config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task_index)])
        with tf.train.MonitoredTrainingSession(master=server.target,
            is_chief=(task_index == 0),
            checkpoint_dir="/tmp/train_logs",
            hooks=hooks,
            config=config) as sess:
            i = 0
            start = time.time()
            done = False
            model.sess = sess
            runner.start()
            runner.start_time=time.time()
            while not sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                # mon_sess.run(train_op)
                runner.process(sess)
                
                

                    
                
                


if __name__ == "__main__":

    ps_hosts = ['localhost:2220']
    worker_hosts = ['localhost:2221']
    worker_hosts = ['localhost:222' + str(i) for i in range(1,8)]

    worker_jobs = [('worker',i) for i in range(len(worker_hosts))]
    ps_jobs = [('ps',i) for i in range(len(ps_hosts))]

    jobs = ps_jobs + worker_jobs 

    for job_name, task_index in jobs:
        print('job_name', job_name)
        print('task_index', task_index)
    
    lock = multiprocessing.Lock()
    #T = multiprocessing.Value('i',0)
    #t0 = multiprocessing.Value('i',0)
   
    ps_procs = [multiprocessing.Process(target=main, args=(ps_hosts,worker_hosts,job_name,task_index), daemon=True) for job_name, task_index in ps_jobs]
    worker_procs = [multiprocessing.Process(target=main, args=(ps_hosts,worker_hosts,job_name,task_index), daemon=True) for job_name, task_index in worker_jobs]
    

    for p in ps_procs:
        p.start()
    
    for p in worker_procs:
        p.start()
    
    for p in worker_procs:
        p.join()





# def main(_):
#   ps_hosts = FLAGS.ps_hosts.split(",")
#   worker_hosts = FLAGS.worker_hosts.split(",")

#   # Create a cluster from the parameter server and worker hosts.
#   cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

#   # Create and start a server for the local task.
#   server = tf.train.Server(cluster,
#                            job_name=FLAGS.job_name,
#                            task_index=FLAGS.task_index)

#   if FLAGS.job_name == "ps":
#     server.join()
#   elif FLAGS.job_name == "worker":

#     # Assigns ops to the local worker by default.
#     with tf.device(tf.train.replica_device_setter(
#         worker_device="/job:worker/task:%d" % FLAGS.task_index,
#         cluster=cluster)):
      
#       model = ActorCritic(Nature_CNN, 6)
#       # Build model...
#       loss = model.loss
#       global_step = tf.contrib.framework.get_or_create_global_step()

#       train_op = tf.train.AdagradOptimizer(0.01).minimize(
#           loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
#     hooks=[tf.train.StopAtStepHook(last_step=1000000)]

#     # The MonitoredTrainingSession takes care of session initialization,
#     # restoring from a checkpoint, saving to a checkpoint, and closing when done
#     # or an error occurs.
#     with tf.train.MonitoredTrainingSession(master=server.target,
#                                            is_chief=(FLAGS.task_index == 0),
#                                            checkpoint_dir="/tmp/train_logs",
#                                            hooks=hooks) as mon_sess:
#       while not mon_sess.should_stop():
#         # Run a training step asynchronously.
#         # See `tf.train.SyncReplicasOptimizer` for additional details on how to
#         # perform *synchronous* training.
#         # mon_sess.run handles AbortedError in case of preempted PS.
#         mon_sess.run(train_op)


# # if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.register("type", "bool", lambda v: v.lower() == "true")
#   # Flags for defining the tf.train.ClusterSpec
#   parser.add_argument(
#       "--ps_hosts",
#       type=str,
#       default="",
#       help="Comma-separated list of hostname:port pairs"
#   )
#   parser.add_argument(
#       "--worker_hosts",
#       type=str,
#       default="",
#       help="Comma-separated list of hostname:port pairs"
#   )
#   parser.add_argument(
#       "--job_name",
#       type=str,
#       default="",
#       help="One of 'ps', 'worker'"
#   )
#   # Flags for defining the tf.train.Server
#   parser.add_argument(
#       "--task_index",
#       type=int,
#       default=0,
#       help="Index of task within the job"
#   )
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)