#!/usr/bin/env python3
"""
Created on 16:39, Feb. 20th, 2023

@author: Norbert Zheng
"""
import gym, wandb
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from threading import Thread
from multiprocessing import cpu_count

# Initialize float precision of keras backend.
K.backend.set_floatx("float64")

# Initialize wandb monitor.
wandb.init(name="A3C-continuous", project="Machine-Learning")

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

# Define global variables.
EPISODE_CURR = 0

# def Actor class
class Actor:
    """
    The actor in A3C model.
    """

    def __init__(self, n_states, n_actions, action_scale, stddev_bound):
        """
        Initialize `Actor` object.

        Args:
            n_states: int - The number of states.
            n_actions: int - The number of actions.
            action_scale: (n_actions,) - The scale of action, mapping [-1,1] to [-scale,scale].
            stddev_bound: (2,) - The bound of standard deviation.

        Returns:
            None
        """
        # Initialize parameters.
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_scale = action_scale
        self.stddev_bound = stddev_bound
        self.entropy_beta = 0.01
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
            model: K.models.Model - The initialized `Actor` model.
        """
        # Initialize the input state.
        state_input = K.layers.Input((self.n_states,))
        # Forward input through MLP.
        mlp_output = K.Sequential([
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(32, activation="relu"),
        ])(state_input)
        # Project `mlp_output` to `mu_output` & `stddev_output`.
        mu_output = K.Sequential([
            K.layers.Dense(self.n_actions, activation="tanh"),
            K.layers.Lambda(lambda x:  x * self.action_scale),
        ])(mlp_output)
        stddev_output = K.layers.Dense(self.n_actions, activation="softplus")(mlp_output)
        # Return the final `model`.
        return K.models.Model(state_input, [mu_output,stddev_output])

    """
    data funcs
    """
    # def get_action func
    def get_action(self, state):
        """
        Get the predicted action of the corresponding state.

        Args:
            state: (batch_size, n_states) - The current state.

        Returns:
            action: (batch_size, n_actions) - The predicted action of the corresponding state.
        """
        mu, stddev = self.model.predict(state)
        return tf.clip_by_value(mu + stddev * tf.random.normal(stddev.shape), -self.action_scale, self.action_scale)

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
        Calculate negative log likelihood of `a_true` under Gaussian distribution specified by `a_pred`.

        Args:
            a_true: (batch_size, n_actions) - The true actions that agent takes.
            a_pred: (mu, stddev) - The predicted actions distribution that agent takes.
                mu: (batch_size, n_actions) - The mean of predicted action Gaussian distribution.
                stddev: (batch_size, n_actions) - The standard deviation of action Gaussian distribution.
            advantages: (batch_size,) - The advantage of each action sample.

        Returns:
            loss: float - The negative log likelihood of `a_true` under Gaussian distribution specified by `a_pred`.
        """
        # Calculate the log pdf of true action from Gaussian distribution specified by (mu, stddev).
        log_pdf = self.log_pdf(a_pred[0], a_pred[1], a_true)
        # Use `advantages` to weight `log_pdf` to get `loss`.
        loss = tf.reduce_sum(-(log_pdf * advantages))
        # Return the final `loss`.
        return loss

    # def log_pdf func
    def log_pdf(self, mu, stddev, action):
        """
        Get log likelihood of action from Gaussian distribution specified by (mu, stddev).

        Args:
            mu: (batch_size, n_actions) - The mean of Gaussian distribution.
            stddev: (batch_size, n_actions) - The standard deviation of Gaussian distribution.
            action: (batch_size, n_actions) - The true action that agent takes.

        Returns:
            log_pdf: (batch_size,) - The log pdf of true action from Gaussian distribution specified by (mu, stddev).
        """
        # Clip `stddev` to avoid outliers.
        stddev = tf.clip_by_value(stddev, self.stddev_bound[0], self.stddev_bound[1]); var = stddev ** 2
        # Calculate the log pdf of true action from Gaussian distribution specified by (mu, stddev).
        log_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        log_pdf = tf.reduce_sum(log_pdf, axis=-1)
        # Return the final `log_pdf`.
        return log_pdf

# def Critic class
class Critic:
    """
    The Critic in A3C model.
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
            K.layers.Dense(32, activation="relu"),
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

# def AgentWorker class
class AgentWorker(Thread):
    """
    The A3C agent worker.
    """

    def __init__(self, env, actor_global, critic_global, n_episodes):
        """
        Initialize `Agent` object.

        Args:
            env: gym - The environment that agent interacts with.
            actor_global: Actor - The global actor.
            critic_global: Critic - The global critic.
            n_episodes: int - The number of training episodes.

        Returns:
            None
        """
        # Initialize super class to inherit `Thread`-style attributes.
        super(AgentWorker, self).__init__()
        # Initialize parameters.
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.action_scale = self.env.action_space.high[0]
        self.stddev_bound = [1e-2, 1.0]
        self.n_episodes = n_episodes
        # Initialize variables.
        self.actor_global = actor_global
        self.critic_global = critic_global
        self.actor_local = Actor(self.n_states, self.n_actions, self.action_scale, self.stddev_bound)
        self.critic_local = Critic(self.n_states)
        # Copy global variables to local variables.
        self.actor_local.model.set_weights(self.actor_global.model.get_weights())
        self.critic_local.model.set_weights(self.critic_global.model.get_weights())

    """
    data funcs
    """
    # def v_true func
    def v_true(self, rewards, state_n, done):
        """
        Get the true values of the corresponding states in sequential order.

        Args:
            rewards: (seq_len,) - The reward of the corresponding states in sequential order.
            state_n: (n_states,) - The one-hot encoding of next state.
            done: bool - The flag indicates whether the corresponding states are goal state.

        Returns:
            values: (seq_len,) - The true values of the corresponding states in sequential order.
        """
        # Initialize `v_true` as an empty list, and unstack `rewards` as a list of tensors.
        v_true = []; rewards = tf.unstack(rewards, axis=0)
        # Initialize `cumulative` according to whether current sequence has been done.
        cumulative = 0 if done else tf.squeeze(self.v_pred(tf.expand_dims(state_n, axis=0)))
        # Fill up `v_true` according to `rewards`.
        for step_idx in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[step_idx]; v_true.append(cumulative)
        v_true.reverse(); v_true = tf.stack(v_true, axis=0)
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
        return self.critic_local.model.predict(states)

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
    def train(self):
        """
        Train `Agent` object.

        Args:
            None

        Returns:
            None
        """
        global EPISODE_CURR
        while self.n_episodes >= EPISODE_CURR:
            # Initialize variables used in one training episode.
            reward_episode = 0.; done = False
            state = tf.expand_dims(tf.constant(self.env.reset()), axis=0)
            state_batch = []; action_batch = []; reward_batch = []

            # Keep interacting with the environment until getting to the goal state.
            while not done:
                # Render the environment.
                #self.env.render()

                # Generate the probability over actions, then execute action.
                action = self.actor_local.get_action(state).numpy()
                state_n, reward, done, _ = self.env.step(action); reward_episode += np.squeeze(reward)
                # Cast `state_n` & `reward` to tensors.
                action = tf.expand_dims(tf.constant(np.squeeze(action), dtype=tf.float64), axis=0)
                state_n = tf.expand_dims(tf.constant(np.squeeze(state_n), dtype=tf.float64), axis=0)
                reward = tf.expand_dims(tf.constant(np.squeeze(reward), dtype=tf.float64), axis=0)

                # Update the corresponding items in `*_batch`.
                state_batch.append(state); action_batch.append(action); reward_batch.append(reward)
                # Check whether `*_batch` has enough batches or we have gotten to goal state.
                if len(state_batch) >= args.update_interval or done:
                    # Stack to get `batch_size`-wise items.
                    states = tf.concat(state_batch, axis=0)
                    actions = tf.concat(action_batch, axis=0)
                    rewards = tf.concat(reward_batch, axis=0)
                    # Calculate the true value and advantage of current state.
                    values = self.v_true((rewards + 8.) / 8., tf.squeeze(state_n), done)
                    advantages = self.advantage(values, tf.squeeze(self.v_pred(states)))
                    # Train actor & critic.
                    loss_actor = self.actor_global.train(states, actions, advantages)
                    loss_critic = self.critic_global.train(states, values)
                    self.actor_local.model.set_weights(self.actor_global.model.get_weights())
                    self.critic_local.model.set_weights(self.critic_global.model.get_weights())
                    # Clear `*_batch`.
                    state_batch = []; action_batch = []; reward_batch = []
                # Update current state.
                state = state_n
            print("Episode {:d}: The accumulated reward of current episode is {:.2f}.".format(EPISODE_CURR, reward_episode))
            wandb.log({"reward":reward_episode,}); EPISODE_CURR += 1

    # def run func
    def run(self):
        """
        Run `AgentWorker` object.

        Args:
            None

        Returns:
            None
        """
        self.train()

# def Agent class
class Agent:
    """
    The A3C agent.
    """

    def __init__(self, env_name):
        """
        Initialize `Agent` object.

        Args:
            env_name: The name of gym environment.

        Returns:
            None
        """
        # Initialize parameters.
        env = gym.make(env_name)
        self.env_name = env_name
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.action_scale = env.action_space.high[0]
        self.stddev_bound = [1e-2, 1.0]
        self.n_workers = cpu_count()
        # Initialize variables.
        self.actor = Actor(self.n_states, self.n_actions, self.action_scale, self.stddev_bound)
        self.critic = Critic(self.n_states)

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
        # Initialize `workers` according to `n_workers`.
        workers = []
        for worker_idx in range(self.n_workers):
            workers.append(AgentWorker(gym.make(self.env_name), self.actor, self.critic, n_episodes))
        # Start parallel training of `Agent`.
        for worker_i in workers: worker_i.start()
        for worker_i in workers: worker_i.join()

# def main func
def main():
    # Initialize env & agent.
    env_name = "Pendulum-v0"
    agent = Agent(env_name)

    # Start the training process of agent.
    agent.train()

if __name__ == "__main__":
    main()

