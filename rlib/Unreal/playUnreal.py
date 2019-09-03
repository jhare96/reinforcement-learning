from rlib.Unreal.UnrealA2C_CNN import*
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
    
    model = UnrealA2C(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      PC=1,
                      entropy_coeff=0.001,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={})

    

    auxiliary = Unreal_Trainer(envs = envs,
                                model = model,
                                val_envs = val_envs,
                                train_mode = 'nstep',
                                total_steps = 50e6,
                                nsteps = nsteps,
                                normalise_obs=True,
                                validate_freq = 0,
                                save_freq = 0,
                                render_freq = 0,
                                num_val_episodes = 1,
                                log_scalars = False,
                                gpu_growth=True)
    
    auxiliary.load_model('10', model_dir)
    
    env = gym.wrappers.Monitor(val_envs[0], "vids/Unreal/" + env_id, force=True)
    auxiliary.validate(env, 1, 10000, True)
    
    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', nargs='?', default='MontezumaRevenge')
    parser.add_argument('--model_dir', nargs='?', default='models/MontezumaRevenge')
    args = parser.parse_args()

    main(args.env + 'Deterministic-v4', args.model_dir)