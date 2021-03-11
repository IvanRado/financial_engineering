import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


class Env:
    def __init__(self, df, feats):
        self.df = df
        self.n = len(df)
        self.current_idx = 0
        self.action_space = [0, 1, 2]  # Buy, Sell, Hold
        self.invested = 0

        self.states = self.df[feats].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()

    def reset(self):
        self.current_idx = 0
        return self.states[self.current_idx]

    def step(self, action):
        # Return (next state, reward, done)
        self.current_idx += 1
        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        if action == 0:
            self.invested = 1
        elif action == 1:
            self.invested = 0

        # Compute reward
        if self.invested:
            reward = self.rewards[self.current_idx]
        else:
            reward = 0

        # Next state transition
        next_state = self.states[self.current_idx]

        # Done flag
        done = (self.current_idx == self.n - 1)
        return next_state, reward, done


class StateMapper:
    def __init__(self, env, n_bins=6, n_samples=10000):
        # First collect sample states from the environment
        states = []
        done = False
        s = env.reset()
        self.D = len(s)  # Number of elements we need to bin
        states.append(s)

        while True:
            a = np.random.choice(env.action_space)
            s2, _, done = env.step(s)
            states.append(s2)
            if len(states) >= n_samples:
                break
            if done:
                s = env.reset()
                states.append(s)
                if len(states) >= n_samples:
                    break

        # Convert to numpy array for easy indexing
        states = np.array(states)

        # Create the bins for each dimension
        self.bins = []
        for d in range(self.D):
            column = np.sort(states[:, d])

            # Find the boundaries for each bin
            current_bin = []
            for k in range(n_bins):
                boundary = column[int(n_samples / n_bins * (k + 0.5))]
                current_bin.append(boundary)

            self.bins.append(current_bin)

    def transform(self, state):
        x = np.zeros(self.D)
        for d in range(self.D):
            x[d] = int(np.digitize(state[d], self.bins[d]))
        return tuple(x)

    def all_possible_states(self):
        list_of_bins = []
        for d in range(self.D):
            list_of_bins.append(list(range(len(self.bins[d]) + 1)))
        return itertools.product(*list_of_bins)


class Agent:
    def __init__(self, action_size, state_mapper):
        self.action_size = action_size
        self.gamma = 0.8  # Discount rate
        self.epsilon = 0.1
        self.learning_rate = 1e-1
        self.state_mapper = state_mapper

        # Initialize Q-table randomly
        self.Q = {}
        for s in self.state_mapper.all_possible_state():
            s = tuple(s)
            for a in range(self.action_size):
                self.Q[(s, a)] = np.random.randn()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        s = self.state_mapper.transform(state)
        act_values = [self.Q[(s, a)] for a in range(self.action_size)]
        return np.argmax(act_values)  # Returns an action

    def train(self, state, action, reward, next_state, done):
        s = self.state_mapper.transform(state)
        s2 = self.state_mapper.transform(next_state)

        if done:
            target = reward
        else:
            act_values = [self.Q(s2, a) for a in range(self.action_size)]
            target = reward + self.gamma * np.amax(act_values)

        # Run one training step
        self.Q[(s, action)] += self.learning_rate * (target - self.Q[(s, action)])


    def act(self, state):
        assert(len(state) == 2)
        # (fast, slow)

        if state[0] > state[1] and not self.is_invested:
            self.is_invested = True
            return 0  # Buy

        if state[0] < state[1] and self.is_invested:
            self.is_invested = False
            return 1  # Sell

        return 2  # Hold


def play_one_episode(agent, env, is_train):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if is_train:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return total_reward


def main():
    df0 = pd.read_csv('sp500_close.csv', index_col=0, parse_dates=True)
    df0.dropna(axis=0, how='all', inplace=True)
    df0.dropna(axis=1, how='any', inplace=True)

    df_returns = pd.DataFrame()
    for name in df0.columns:
        df_returns[name] = np.log(df0[name]).diff()

    Ntest = 1000
    train_data = df_returns.iloc[:-Ntest]
    test_data = df_returns.iloc[-Ntest:]

    feats = ['AAPl', 'MSFT', 'AMZN']
    num_episodes = 500

    train_env = Env(train_data, feats)
    test_env = Env(test_data, feats)

    action_size = len(train_env.action_space)
    state_mapper = StateMapper(train_env)
    agent = Agent(action_size, state_mapper)

    train_rewards = np.empty(num_episodes)
    test_rewards = np.empty(num_episodes)

    for e in range(num_episodes):
        r = play_one_episode(agent, train_env, is_train=True)
        train_rewards[e] = r

        # Test on the test set
        tmp_epsilon = agent.epsilon
        agent.epsilon = 0
        tr = play_one_episode(agent, test_env, is_train=False)
        agent.epsilon = tmp_epsilon
        test_rewards[e] = tr

        print(f"Episode: {e + 1}/{num_episodes}, train: {r:.5f}, test: {tr:.5f}")
        plt.plot(train_rewards)
        plt.plot(test_rewards)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
