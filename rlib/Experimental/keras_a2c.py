import numpy as np
import tensorflow as tf
import time, datetime
import gym
from VecEnv import BatchEnv, Env, AtariEnv, StackEnv
#tf.enable_eager_execution()





def validate(model,env,num_ep,max_steps,render=False):
    episode_scores = []
    for episode in range(num_ep):
        state = env.reset()
        episode_score = []
        for t in range(max_steps):
            policy, value = model(state[np.newaxis,:])
            #policy = tf.keras.backend.get_value(policy)
            policy = policy.numpy()
            action = np.random.choice(policy.shape[1], p=policy[0])
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if render:
                env.render()

            episode_score.append(reward)

            if done or t == max_steps -1:
                tot_reward = np.sum(episode_score)
                episode_scores.append(tot_reward)
                break
    
    avg_score = np.mean(episode_scores)
    return avg_score


def loss(model, x, y, actions, action_size):
    policy_distrib, value = model(x)
    actions_onehot = tf.keras.backend.one_hot(actions, action_size)
    Advantage = y - value
    value_loss = tf.keras.backend.square(Advantage)
    log_policy = tf.keras.backend.log(policy_distrib)
    log_policy_actions = tf.keras.backend.sum(log_policy * actions_onehot, axis=1)
    policy_loss =  -log_policy_actions * tf.stop_gradient(Advantage)
    entropy =  tf.keras.backend.sum(policy_distrib * -log_policy, axis=1)
    loss =  tf.keras.backend.sum( policy_loss + 0.5 * value_loss - 0.01 * entropy)
    return loss 

#@tf.function
def loss2(model, x, y, actions, action_size):
    policy_distrib, value = model(x)
    actions_onehot = tf.keras.backend.one_hot(actions, action_size)
    Advantage = y - value
    value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))
    log_policy = tf.math.log(tf.clip_by_value(policy_distrib, 1e-6, 0.999999))
    log_policy_actions = tf.reduce_sum(log_policy * actions_onehot, axis=1)
    policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))
    entropy =  tf.reduce_mean(tf.reduce_sum(policy_distrib * -log_policy, axis=1))
    loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy
    return loss


def train(model, optimiser, env, val_env, nsteps, total_steps, num_envs, validate_freq):
    #score = validate(model,val_env,5,5000,render=False)
    states_ = env.reset()
    num_updates = total_steps // (num_envs * nsteps)
    start = time.time()
    for update in range(1,num_updates+1):
        memory = []
        for t in range(nsteps):
            policies, values = model(states_)
            #values = tf.keras.backend.get_value(values)
            #values = np.reshape(values, [-1])
            policies = policies.numpy()
            #actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
            actions = np.argmax(policies, 1)
            next_states, rewards, dones, infos = env.step(actions)
            memory.append((states_, actions, rewards, dones, infos))
            states_ = next_states
        
        states, actions, rewards, dones, infos = zip(*memory)
        
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        #policies = np.stack(policies)#tf.keras.backend.stack(policies)
        #values = np.stack(values)#tf.keras.backend.stack(values)
        rewards = np.clip(rewards, -1, 1)
        T = len(rewards)
        
        # Calculate R for advantage A = R - V 
        R = np.zeros((T,num_envs))
        v = values.numpy()
        v = v.reshape(-1)
        
        R[-1] = v * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as wrapped env continues after end of episode
            R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
            
        # stack all states, actions and Rs into a single batch
        #policies, values = fold_keras(policies), fold_keras(values)
        states, actions, R,  = fold_batch(states), fold_batch(actions,dtype=np.int32), fold_batch(R)
        
        #start2 = time.time()
        with tf.GradientTape() as tape:
            #loss_value = loss(model, states, tf.convert_to_tensor(R), tf.convert_to_tensor(actions), 6)
            loss_value = loss2(model,states, R, actions, 6)
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        optimiser.apply_gradients(zip(grads, model.trainable_variables))
        #print('backprop time', time.time()-start2)
    
    
    
        # if self.render_freq > 0 and update % (self.validate_freq * self.render_freq) == 0:
        #     render = True
        # else:
        #     render = False

        if update % validate_freq == 0:
            tot_steps = update * num_envs * nsteps
            time_taken = time.time() - start
            frames_per_update = (validate_freq * num_envs * nsteps)
            fps = frames_per_update / time_taken 
            score = validate(model,val_env,5,5000,render=False)
            print("update %i, validation score %f, total steps %i, loss %f, time taken for %i frames:%f, fps %f" %(update,score,tot_steps,loss_value,frames_per_update,time_taken,fps))
            # sumscore, sumloss = self.sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:score, tf_epLoss:l})
            # self.train_writer.add_summary(sumloss, tot_steps)
            # self.train_writer.add_summary(sumscore, tot_steps)
            start = time.time()
            
            # if self.save_freq > 0 and  update % self.save_freq == 0:
            #     s += 1
            #     self.saver.save(self.sess, str(self.model_dir + self.modelname + '_' + str(s) + ".ckpt") )
#@tf.function
def fold_keras(x):
    y = tf.keras.backend.reshape(x, [20,-1])
    return y

def fold_batch(x,dtype=np.float32):
    rows, cols = x.shape[0], x.shape[1]
    y = x.reshape(rows*cols,*x.shape[2:]).astype(dtype)
    return y


class CNN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32,[8,8],[4,4],'VALID',activation=tf.keras.activations.relu)
        self.conv2 = tf.keras.layers.Conv2D(64,[4,4],[2,2],'VALID',activation=tf.keras.activations.relu)
        self.conv3 = tf.keras.layers.Conv2D(64,[3,3],[1,1],'VALID',activation=tf.keras.activations.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)
        self.policy = tf.keras.layers.Dense(action_size, activation=tf.keras.activations.softmax)
        self.value = tf.keras.layers.Dense(1, activation=None)
    
    def call(self,state):
        x = state / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value



if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


    action_size = 6
    num_workers = 32
    model = CNN()
    optmiser = tf.keras.optimizers.RMSprop(1e-4,0.9, epsilon=1e-5)

    env = BatchEnv(AtariEnv,'SpaceInvaders-v0', num_workers, False, k=4)
    val_env = StackEnv(gym.make('SpaceInvaders-v0'))

    train(model,optmiser,env,val_env,5,int(10e6),num_workers,(16000//num_workers))