import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    Q[s, a] += alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if np.random.rand() < epsilone:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode=None)

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.2 # choose your own

    gamma = 0.95 # choose your own

    epsilon = 1.0  # choose your own
    epsilon_min = 0.01
    epsilon_decay = 0.995

    n_epochs = 10000 # choose your own
    max_itr_per_epoch = 100 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs
    plt.plot(rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Training Rewards')
    plt.show()

    print("Training finished.\n")

    
    """
    Evaluate the q-learning algorithm
    """
    print("Evaluating...")
    eval_rewards = []
    for e in range(10):
        S, _ = env.reset()
        r = 0
        done = False
        while not done:
            A = np.argmax(Q[S])  # greedy policy
            Sprime, R, done, _, _ = env.step(A)
            r += R
            S = Sprime
        eval_rewards.append(r)
    print("Average evaluation reward = ", np.mean(eval_rewards))

    env.close()
