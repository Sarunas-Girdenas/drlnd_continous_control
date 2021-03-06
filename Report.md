### Continuous Control

In this project the task was to solve Reacher environment by controling an artifical hand and moving it to desired location. I've chosen to work with the first version of the environment that has only 1 arm.

For this purpose, I've used [this](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) DDPG agent implementation with slightly modified configuration. For more information about DDPG approach, see Readme file of this repository.

In terms of Actor-Critic, I've tried a few configurations. For example, using [weight normalization](https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html) and [batch normalization](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html). For this project batch normalization for the first layer only seemed to work best. Weight normalization did not improve learning therefore it was removed altogether from both networks. In terms of structure, both networks have 2 layers (128, 128). I've also tried different depth (deep Actor and Shallow Critic) but this did not improve the convergence as well.

In terms of hyperparameters used, they are mostly taken from the existing implimentations and have the following values: </br>

`BUFFER_SIZE = int(1e6)  # replay buffer` size <br/>
`BATCH_SIZE = 128        # minibatch size` <br/>
`GAMMA = 0.99            # discount factor` <br/>
`TAU = 1e-3              # for soft update of target parameters` <br/>
`LR_ACTOR = 2e-4         # learning rate of the actor` <br/>
`LR_CRITIC = 2e-4        # learning rate of the critic` <br/>
`WEIGHT_DECAY = 0        # L2 weight decay` <br/>

During the training I've also tried changing the batch size. Larger values, such as 1024 or 512 seemed not to perform too well (agent took a very long time to reach any score above 2) whereas size of 128 was a really good choice based on learning speed. Both Actor and Critic have the same learning rates, this seemed not to affect training too much. However, any `weight_decay` value larger than 0 seemed to be detrimental to learning process. One explanation for this could be the fact that having non-zero value reguralizes Critic network too much and as a result its not able to learn anyting meaningful. Another option would be to increase depth/width of the Critic network and then try having `weight_decay` larger than 0; this could potentialy improve learning of the Agent.

I've also tried implementing Prioritised Experience Replay to improve learning but I was unable to find the right configuration for it to work. It either did not train at all (reward was fluctuating between 1-2 for more than  500 episodes) or it was crashing due to unsuccessful experience sampling. For completeness, I've included the file in the repository, it is called `ddpg_agent_per.py`.

While training the agent, I've observed that during some runs training will stop at around 25 reward points and after that it will reduce, i.e. subsequent episodes will result in worse performance. This probably happened because optimizer has stuck in sub-optimal point. Restarting the training (and not changing any parameters) seemed to solve this problem.

There are a few possible improvements that could improve training:
1. Make Prioritised Experience Replay work. This should result in faster training as experiences with larger errors would be sampled more often.
2. Use gradient clipping for Actor network (suggested in the benchmark implementation). I've added it to the training code and it seemd to improve the convergence.
3. Initializing Actor and Critic local and target weights to the same values. This seemed to give the biggest improvement in the early episodes ofx training.
4. Investigate other architectures of Actor and Critic networks. For example, increase the depth and add Dropout or L2 normalization to optimizers.