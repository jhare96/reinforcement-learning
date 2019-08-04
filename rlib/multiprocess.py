import numpy as np 
import gym 
import multiprocessing as mp
import time
a = 1


def generate_ep(q,env,wait,locked,worker_id,master=False):
    #env = gym.make(id)
    state = env.reset()
    
    for ep in range(30):
        actions = []
        rewards = []
        states = []
        for t in range(10000):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            actions.append(action)
            rewards.append(reward)
            states.append(states)
            env.render()

            if done or t == 10000-1:
                env.reset()
                q.put((states,actions,rewards))
                break
        
        
        #wait[worker_id] = 1
        # while locked[0]:
        #     if any(thread == 0 for thread in wait):
        #         print("worker ID", worker_id)
        #         print("wait", wait[:])
        #         time.sleep(5)
        #     elif master:
        #         for i in range(15,-1,-1):
        #             print("i",i)
        #             wait[i] = 0
        #         locked[0] = 0
        
            
    #env.close()
    #return (states,actions,rewards)


def main():
    id = "SpaceInvaders-v0"
    q = mp.Queue()
    processes = []
    num_procs = 8
    wait = mp.Array('i', num_procs)
    locked = mp.Array('i', 1)
    locked[0] = True
    
    #pool = mp.Pool(processes=16)
    envs = [gym.make(id) for i in range(num_procs)]
    start = time.time()
    #results = [pool.apply_async(generate_ep, args=(env,)) for env in envs]
    #rollouts = [p.get() for p in results]
    for i in range(num_procs):
        if i == 0:
            master = True
        else:
            master = False
        p = mp.Process(target=generate_ep, args=(q,envs[i],wait,locked,i,master))
        processes.append(p)
        
    try:
        for p in processes:
            p.start()
        
        while True:
            rollouts = q.get()
            print("rollout length", len(rollouts[0]))
        
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("parent crtl C recieved")
        for p in processes:
            #p.daemon = True
            p.terminate()
            p.join()
        exit()

    

    print("time taken", time.time()-start)

if __name__ == "__main__":
    main()
    