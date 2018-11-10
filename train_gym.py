







if __name__ == '__main__':
    update_every = 4
    gamma = 0.99 # discount factor
    tau=0.01 # soft update rate

    # exploration
    exploration_mu = 0.0
    exploration_theta = 0.15
    exploration_sigma = 0.20

    # replay buffer
    buffer_size = 1e5
    batch_size = 64