# reinforcement-learning

A small Pytorch based reinforcement learning library  
as used for my MSc dissertation project ['Dealing with sparse rewards in reinforcement learning'](https://arxiv.org/abs/1910.09281).  

This repository has working implementations of the following reinforcement agents:  
          1. Advantage Actor Critic [(A2C)](https://openai.com/blog/baselines-acktr-a2c/)  
          2. Synchronous n-step Double Deep Q Network (Sync-DDQN)  
          3. Proximal Policy Optimisation [(PPO)](https://arxiv.org/abs/1707.06347)  
          4. Random Network Distillation [(RND)](https://arxiv.org/abs/1810.12894)
          5. UNREAL-A2C2, A2C-CNN version of the [(UNREAL agent)](https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks)  
          6. Random Network Distillation with Auxiliary Learning (RANDAL), novel solution combining UNREAL and RND agents  
          
          
# Install repository:
```bash
git clone https://github.com/jhare96/reinforcement-learning.git
pip install -e reinforcement-learning
```
# To cite RANDAL agents in publications:  
follow the link to the ArXiv publication https://arxiv.org/abs/1910.09281
          
# To cite this repository in publications:

    @misc{Hare_rlib,
      author = {Joshua Hare},
      title = {reinforcement learning library, rlib},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/jhare96/reinforcement-learning}},
    }

