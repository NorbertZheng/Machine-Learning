#!/usr/bin/env python3
"""
Created on 12:13, Feb. 20th, 2023

@author: Norbert Zheng
"""
import gym, wandb
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

# Initialize float precision of keras backend.
K.backend.set_floatx("float64")

# Initialize wandb monitor.
wandb.init(name="A2C-discrete", project="Machine-Learning")

## Initialize parser.
parser = argparse.ArgumentParser()
# The reward decay parameter.
parser.add_argument("--gamma", type=float, default=0.99)
# The number of accumulated batches.
parser.add_argument("--update_interval", type=int, default=5)
# The learning rate of actor.
parser.add_argument("--lr_actor", type=float, default=5e-4)
# The learning rate of critic.
parser.add_argument("--lr_critic", type=float, default=1e-3)
args = parser.parse_args()

# def Actor class
class Actor:
    """
    The actor in A2C model.
    """

    def __init__(self, n_states, n_actions):
        """
        Initialize `Actor` object.
        
        Args:
            n_states: int - The number of states.
            n_actions: int - The number of actions.

        Returns:
            None
        """
        # Initialize parameters.
        self.n_states = n_states
        self.n_actions = n_actions
        # Initialize variables.
        self.model = self._init_model()
        self.optimizer = K.optimizers.Adam(learning_rate=args.lr_actor)

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize `Actor` model.

        Args:
            None

        Returns:
            model: K.Sequential - The initialized `Actor` model.
        """
        return K.Sequential([
            K.layers.Input((self.n_states,)),
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(16, activation="relu"),
            K.layers.Dense(self.n_actions, activation="softmax"),
        ])

    """
    train funcs
    """
    # def train func
    def train(self, states, actions, advantages):
        """
        Train `Actor` model.

        Args:
            states: (batch_size, n_states) - The input states.
            actions: (batch_size, n_actions) - The true actions that agent takes at the corresponding states.
            advantages: (batch_size,) - The advantage of each (state, action) sample pair.

        Returns:
            loss: float - The cross entropy loss between true actions and predicted actions.
        """
        with tf.GradientTape() as gt:
            a_pred = self.model(states, training=True)
            loss = self.loss(actions, a_pred, advantages)
        grads = gt.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Return the final `loss`.
        return loss

    """
    loss funcs
    """
    # def loss func
    def loss(self, a_true, a_pred, advantages):
        """
        Calculate cross entropy loss between a_true (true) and a_pred (predicted), weighted by advantages.

        Args:
            a_true: (batch_size, n_actions) - The true actions that agent takes.
            a_pred: (batch_size, n_actions) - The predicted actions that agent takes.
            advantages: (batch_size,) - The advantage of each action sample.

        Returns:
            loss: float - The cross entropy loss between a_true (true) and a_pred (predicted).
        """
        # Cast `a_true` to `tf.int64`.
        a_true = tf.cast(a_true, dtype=tf.int64)
        # Calculate cross entropy loss.
        loss = K.losses.SparseCategoricalCrossentropy(from_logits=True)(
            a_true, a_pred, sample_weight=tf.stop_gradient(advantages))
        # Return the final `loss`.
        return loss

# def Critic class
class Critic:
    """
    The Critic in A2C model.
    """

    def __init__(self, n_states):
        """
        Initialize `Critic` object.

        Args:
            n_states: The number of states.

        Returns:
            None
        """
        # Initialize parameters.
        self.n_states = n_states
        # Initialize variables.
        self.model = self._init_model()
        self.optimizer = K.optimizers.Adam(learning_rate=args.lr_critic)

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize `Critic` model.

        Args:
            None

        Returns:
            model: K.Sequential - The initialized `Actor` model.
        """
        return K.Sequential([
            K.layers.Input((self.n_states,)),
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(16, activation="relu"),
            K.layers.Dense(16, activation="relu"),
            K.layers.Dense(1, activation="linear"),
        ])

    """
    train funcs
    """
    # def train func
    def train(self, states, values):
        """
        Train `Actor` model.

        Args:
            state: (batch_size, n_states) - The input state.
            value: (batch_size,) - The true value of the corresponding state.

        Returns:
            loss: float - The mean square error loss between actions (true) and logits (predicted).
        """
        with tf.GradientTape() as gt:
            v_pred = self.model(states, training=True)
            loss = self.loss(tf.stop_gradient(values), v_pred)
        grads = gt.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Return the final `loss`.
        return loss

    """
    loss funcs
    """
    # def loss func
    def loss(self, v_true, v_pred):
        """
        Calculate mean square error loss between v_true (true) and v_pred (predicted).

        Args:
            v_true: (batch_size,) - The true value of the corresponding state.
            v_pred: (batch_size,) - The predicted value of the corresponding state.

        Returns:
            loss: float - The mean square error loss between v_true (true) and v_pred (predicted).
        """
        return K.losses.MeanSquaredError()(v_true, v_pred)

# def Agent class
class Agent:
    """
    The A2C agent.
    """

    def __init__(self, env):
        """
        Initialize `Agent` object.

        Args:
            env: gym - The environment that agent interacts with.

        Returns:
            None
        """
        # Initialize parameters.
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        # Initialize variables.
        self.actor = Actor(self.n_states, self.n_actions)
        self.critic = Critic(self.n_states)

    """
    data funcs
    """
    # def v_true func
    def v_true(self, rewards, states_n, done):
        """
        Get the true values of the corresponding states.

        Args:
            rewards: (batch_size,) - The reward of the corresponding states.
            states_n: (batch_size, n_states) - The one-hot encoding of next states.
            done: (batch_size,) - The flag indicates whether the corresponding states are goal state.

        Returns:
            values: (batch_size,) - The true values of the corresponding states.
        """
        # If the corresponding states are goal state, return rewards as values. Otherwise, use reward plus decayed value.
        v_pred_n = tf.squeeze(self.critic.model.predict(states_n))
        v_true = tf.where(done, rewards, rewards + args.gamma * v_pred_n)
        # Return the final `values`.
        return v_true

    # def v_pred func
    def v_pred(self, states):
        """
        Get the predicted values of the corresponding states.

        Args:
            states: (batch_size, n_states) - The one-hot encoding of the corresponding states.

        Returns:
            values: (batch_size,) - The predicted values of the corresponding states.
        """
        return self.critic.model.predict(states)

    # def advantage func
    def advantage(self, v_true, v_pred):
        """
        Calculate the advantage according to v_true and v_pred.

        Args:
            v_true: (batch_size,) - The true value of current state.
            v_pred: (batch_size,) - The predicted value of current state.

        Returns:
            advantage: (batch_size,) - The advantage of current state.
        """
        return v_true - v_pred

    """
    train funcs
    """
    # def train func
    def train(self, n_episodes=1000):
        """
        Train `Agent` object.

        Args:
            n_episodes: The number of training episodes.

        Returns:
            None
        """
        for episode_idx in range(n_episodes):
            # Initialize variables used in one training episode.
            reward_episode = 0.; done = False
            state = tf.expand_dims(tf.constant(self.env.reset()), axis=0)
            state_batch = []; action_batch = []; value_batch = []; advantage_batch = []

            # Keep interacting with the environment until getting to the goal state.
            while not done:
                # Render the environment.
                #self.env.render()

                # Generate the probability over actions, then execute action.
                action = self.actor.model.predict(tf.reshape(state, [1, self.n_states]))
                action = tf.squeeze(tf.random.categorical(logits=action, num_samples=1)).numpy()
                state_n, reward, done, _ = self.env.step(action); reward_episode += np.squeeze(reward)
                # Cast `state_n` & `reward` to tensors.
                action = tf.expand_dims(tf.constant(np.squeeze(action), dtype=tf.float64), axis=0)
                state_n = tf.expand_dims(tf.constant(np.squeeze(state_n), dtype=tf.float64), axis=0)
                reward = tf.expand_dims(tf.constant(np.squeeze(reward), dtype=tf.float64), axis=0)
                done_ = tf.expand_dims(tf.constant(np.squeeze(done), dtype=tf.bool), axis=0)
                # Calculate the true value and advantage of current state.
                v_true = self.v_true(reward * 0.01, state_n, done_)
                advantage = self.advantage(v_true, self.v_pred(state))

                # Update the corresponding items in `*_batch`.
                state_batch.append(state); action_batch.append(action)
                value_batch.append(v_true); advantage_batch.append(advantage)
                # Check whether `*_batch` has enough batches or we have gotten to goal state.
                if len(state_batch) >= args.update_interval or done:
                    # Stack to get `batch_size`-wise items.
                    states = tf.concat(state_batch, axis=0); actions = tf.concat(action_batch, axis=0)
                    values = tf.concat(value_batch, axis=0); advantages = tf.concat(advantage_batch, axis=0)
                    # Train actor & critic.
                    _ = self.actor.train(states, actions, advantages); _ = self.critic.train(states, values)
                    # Clear `*_batch`.
                    state_batch = []; action_batch = []; value_batch = []; advantage_batch = []
                # Update current state.
                state = state_n
            print("Episode {:d}: The accumulated reward of current episode is {:.2f}.".format(episode_idx, reward_episode))
            wandb.log({"reward":reward_episode,})

# def main func
def main():
    # Initialize env & agent.
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = Agent(env)

    # Start the training process of agent.
    agent.train()

if __name__ == "__main__":
    main()

