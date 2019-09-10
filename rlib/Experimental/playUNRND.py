from rlib.Experimental.UnRND import*
import argparse




def main(env_id, model_dir):
    num_envs = 1
    nsteps = 128

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(1)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    model = RND(nature_cnn,
                predictor_cnn,
                input_shape = input_size,
                action_size = action_size,
                intr_coeff=1.0,
                extr_coeff=2.0,
                value_coeff=0.5,
                lr=1e-4,
                grad_clip=0.5,
                policy_args={},
                RND_args={}) #

    

    curiosity = RND_Trainer(envs = envs,
                            model = model,
                            val_envs = val_envs,
                            train_mode = 'nstep',
                            total_steps = 50e6,
                            nsteps = nsteps,
                            init_obs_steps=128*50,
                            num_epochs=4,
                            num_minibatches=4,
                            validate_freq = 1e6,
                            save_freq = 0,
                            render_freq = 0,
                            num_val_episodes = 50,
                            log_scalars=False,
                            gpu_growth=True)
    
    curiosity.load_model('3', model_dir)
    
    env = gym.wrappers.Monitor(val_envs[0], "vids/UnRND/" + env_id, force=True)
    curiosity.validate(env, 1, 10000, True)
    
    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', nargs='?', default='MontezumaRevenge')
    parser.add_argument('--modeldir', nargs='?', default='models/MontezumaRevenge')
    args = parser.parse_args()

    main(args.env + 'Deterministic-v4', args.modeldir)