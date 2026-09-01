"""Rainbow DQN optimization core used by speed-policy training."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .network import Network
from .replayBuffer import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    """Categorical Double DQN with dueling NoisyNet, PER, and n-step replay.

    The high-level, simulator-specific loop lives in ``speed_training.py``.
    This class owns action selection, transition collection, and optimizer
    updates so it can also be reused by short integration checks.
    """

    def __init__(
        self,
        env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        lr: float = 1e-4,
        gamma: float = 0.97,
        tau: float = 0.5,
        frame_skip: int = 10,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.1,
        hard_exploration_steps: int = 0,
        exploration_steps: int = 0,
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        v_min: float = 0.0,
        v_max: float = 120.0,
        atom_size: int = 121,
        n_step: int = 3,
        n_step_alpha: float = 1.0,
        lql_trajectory_length: int = 0,
        lql_lambda_lb: float = 1.0,
        lql_lambda_ub: float = 1.0,
        hidden_dim: int = 256,
        device=None,
        **legacy_options,
    ):
        # Logging/checkpoint options belonged to the unrecovered monolithic
        # trainer. Accept them for source compatibility; public loops own I/O.
        ignored = {"log_dir", "file_path", "name", "ckpt_save_freq"}
        unknown = set(legacy_options).difference(ignored)
        if unknown:
            raise TypeError(f"Unexpected DQN options: {', '.join(sorted(unknown))}")
        if batch_size <= 0 or memory_size < batch_size:
            raise ValueError("memory_size must be at least a positive batch_size")
        if frame_skip <= 0 or atom_size < 2 or v_max <= v_min:
            raise ValueError("Invalid frame skip or categorical support")

        random.seed(seed)
        np.random.seed(seed)
        self.env = env
        self.frame_skip = int(frame_skip)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.exploration_steps = int(exploration_steps)
        self.hard_exploration_steps = int(hard_exploration_steps)
        self.min_epsilon = float(min_epsilon)
        self.beta = float(beta)
        self.prior_eps = float(prior_eps)

        obs_dim = int(env.obs_space)
        action_dim = int(env.action_space)
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        self.use_n_step = n_step > 1
        self.n_step = int(n_step)
        self.n_step_alpha = float(n_step_alpha)
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                obs_dim,
                memory_size,
                batch_size,
                n_step=n_step,
                gamma=gamma,
            )

        self.lql_trajectory_length = int(lql_trajectory_length)
        self.lql_lambda_lb = float(lql_lambda_lb)
        self.lql_lambda_ub = float(lql_lambda_ub)
        if self.lql_trajectory_length < 0:
            raise ValueError("lql_trajectory_length must be nonnegative")
        if self.lql_lambda_lb < 0 or self.lql_lambda_ub < 0:
            raise ValueError("LQL hinge weights must be nonnegative")
        self.lql_current_trajectory = []
        self.lql_trajectories = deque(maxlen=int(memory_size))
        self.last_update_stats = {
            "td_loss": 0.0,
            "lql_lb_loss": 0.0,
            "lql_ub_loss": 0.0,
            "lql_lb_active_fraction": 0.0,
            "lql_ub_active_fraction": 0.0,
        }

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atom_size = int(atom_size)
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        self.dqn = Network(
            obs_dim, action_dim, atom_size, self.support, hidden_dim=hidden_dim
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, atom_size, self.support, hidden_dim=hidden_dim
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.transition = []
        self.is_test = False

    def select_action(self, state: np.ndarray) -> int:
        """Select a discrete speed action and start a replay transition."""

        explore = not self.is_test and np.random.uniform() < self.epsilon
        if explore:
            action = int(np.random.randint(self.env.action_space))
        else:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(self.dqn(state_tensor).argmax(dim=1).item())
        if not self.is_test:
            self.transition = [np.asarray(state, dtype=np.float32).copy(), action]
        return action

    def step(self, action: int, frame_skip: int | None = None):
        """Execute one decision-level speed action and store one transition."""

        repeats = self.frame_skip if frame_skip is None else int(frame_skip)
        if repeats <= 0:
            raise ValueError("frame_skip must be positive")
        started = perf_counter()
        step_decision = getattr(self.env, "step_decision", None)
        if step_decision is not None:
            next_state, total_reward, done, info = step_decision(
                action, frame_skip=repeats
            )
        else:
            total_reward = 0.0
            info = {"success": False}
            for _ in range(repeats):
                next_state, reward, done, info = self.env.step(action)
                total_reward += float(reward)
                if done:
                    break
        env_step_time = perf_counter() - started

        buffer_started = perf_counter()
        if not self.is_test:
            if len(self.transition) != 2:
                raise RuntimeError("Call select_action() before step() while training")
            transition = self.transition + [total_reward, next_state, done]
            if self.lql_trajectory_length > 0:
                self.lql_current_trajectory.append(
                    (
                        np.asarray(transition[0], dtype=np.float32).copy(),
                        int(transition[1]),
                        float(transition[2]),
                        np.asarray(transition[3], dtype=np.float32).copy(),
                        bool(transition[4]),
                    )
                )
                if done:
                    self.lql_trajectories.append(tuple(self.lql_current_trajectory))
                    self.lql_current_trajectory = []
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*transition)
            else:
                one_step_transition = transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
        buffer_time = perf_counter() - buffer_started
        result_info = dict(info)
        result_info.update(
            env_step_time=env_step_time,
            add_to_buffer_time=buffer_time,
        )
        return next_state, total_reward, done, result_info

    def update_model(self) -> float:
        """Perform one prioritized categorical DQN update."""

        samples = self.memory.sample_batch(self.beta)
        weights = torch.as_tensor(
            samples["weights"].reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        indices = samples["indices"]
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        if self.use_n_step:
            n_step_samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss = elementwise_loss + self.n_step_alpha * self._compute_dqn_loss(
                n_step_samples, self.gamma**self.n_step
            )
        td_loss = torch.mean(elementwise_loss * weights)
        lql_lb, lql_ub, lql_stats = self._compute_lql_loss()
        loss = (
            td_loss
            + self.lql_lambda_lb * lql_lb
            + self.lql_lambda_ub * lql_ub
        )

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        priorities = elementwise_loss.detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(indices, priorities)
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
        self.last_update_stats = {
            "td_loss": float(td_loss.detach().item()),
            "lql_lb_loss": float(lql_lb.detach().item()),
            "lql_ub_loss": float(lql_ub.detach().item()),
            **lql_stats,
        }
        return float(loss.item())

    def _sample_lql_trajectory(self):
        """Sample one contiguous, within-episode replay chunk."""

        eligible = [
            trajectory for trajectory in self.lql_trajectories
            if len(trajectory) >= 2
        ]
        if not eligible:
            return None
        trajectory = random.choice(eligible)
        length = min(self.lql_trajectory_length, len(trajectory))
        start = random.randint(0, len(trajectory) - length)
        return trajectory[start : start + length]

    def _compute_lql_loss(self):
        """Paper-style two-sided n-step inequality penalties.

        Values and rewards are divided by the categorical support width before
        applying the squared hinge. This is an algebraically equivalent value
        rescaling that keeps lambda=1 commensurate with Rainbow's categorical
        cross-entropy rather than an unnormalized squared-return loss.
        """

        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        empty_stats = {
            "lql_lb_active_fraction": 0.0,
            "lql_ub_active_fraction": 0.0,
        }
        if self.lql_trajectory_length <= 0:
            return zero, zero, empty_stats
        chunk = self._sample_lql_trajectory()
        if chunk is None:
            return zero, zero, empty_stats

        states_np = [transition[0] for transition in chunk]
        states_np.append(chunk[-1][3])
        states = torch.as_tensor(
            np.asarray(states_np), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [transition[1] for transition in chunk],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [transition[2] for transition in chunk],
            dtype=torch.float32,
            device=self.device,
        )
        terminal = bool(chunk[-1][4])
        length = len(chunk)
        scale = max(self.v_max - self.v_min, 1.0)

        online_logged = self.dqn(states[:-1]).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1) / scale
        with torch.no_grad():
            greedy = self.dqn(states).argmax(dim=1)
            target_greedy = self.dqn_target(states).gather(
                1, greedy.unsqueeze(1)
            ).squeeze(1) / scale
            if terminal:
                target_greedy[-1] = 0.0
        scaled_rewards = rewards / scale

        # Prefix returns G[i, j] for all 0 <= i <= j <= L.
        returns = torch.zeros(
            (length + 1, length + 1),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(length):
            discount = 1.0
            for j in range(i + 1, length + 1):
                returns[i, j] = returns[i, j - 1] + discount * scaled_rewards[j - 1]
                discount *= self.gamma

        lb_violations = []
        for k in range(length):
            # Exclude the one-step case, as recommended in the paper.
            for later in range(k + 2, length + 1):
                candidate = returns[k, later] + (
                    self.gamma ** (later - k)
                ) * target_greedy[later]
                lb_violations.append(torch.relu(candidate - online_logged[k]))

        ub_violations = []
        for k in range(1, length):
            # Include i=k (same-state upper bound) plus earlier states.
            for earlier in range(0, k + 1):
                candidate = returns[earlier, k] + (
                    self.gamma ** (k - earlier)
                ) * online_logged[k]
                ub_violations.append(torch.relu(candidate - target_greedy[earlier]))

        def aggregate(values):
            if not values:
                return zero, 0.0
            stacked = torch.stack(values)
            return stacked.square().mean(), float((stacked > 0).float().mean().item())

        lb_loss, lb_active = aggregate(lb_violations)
        ub_loss, ub_active = aggregate(ub_violations)
        return lb_loss, ub_loss, {
            "lql_lb_active_fraction": lb_active,
            "lql_ub_active_fraction": ub_active,
        }

    def decay_epsilon(self, step: int):
        if step <= self.hard_exploration_steps:
            self.epsilon = 1.0
        elif self.exploration_steps <= 0:
            self.epsilon = 0.0
        elif step >= self.exploration_steps:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def set_eval(self, enabled=True):
        self.is_test = bool(enabled)
        self.dqn.eval() if enabled else self.dqn.train()

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(
            samples["next_obs"], dtype=torch.float32, device=self.device
        )
        action = torch.as_tensor(samples["acts"], dtype=torch.long, device=self.device)
        reward = torch.as_tensor(
            samples["rews"].reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(
            samples["done"].reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)[
                range(self.batch_size), next_action
            ]
            target_support = (
                reward + (1 - done) * gamma * self.support
            ).clamp(self.v_min, self.v_max)
            projection = (target_support - self.v_min) / delta_z
            lower = projection.floor().long()
            upper = projection.ceil().long()
            lower[(upper == lower) & (upper > 0)] -= 1
            upper[(upper == lower) & (lower < self.atom_size - 1)] += 1
            offset = (
                torch.arange(self.batch_size, device=self.device).unsqueeze(1)
                * self.atom_size
            )
            projected_dist = torch.zeros_like(next_dist)
            projected_dist.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (next_dist * (upper.float() - projection)).view(-1),
            )
            projected_dist.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (next_dist * (projection - lower.float())).view(-1),
            )

        distribution = self.dqn.dist(state)
        log_probability = torch.log(
            distribution[range(self.batch_size), action].clamp_min(1e-8)
        )
        return -(projected_dist * log_probability).sum(1)

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _target_soft_update(self):
        with torch.no_grad():
            for target, source in zip(
                self.dqn_target.parameters(), self.dqn.parameters()
            ):
                target.copy_(self.tau * source + (1.0 - self.tau) * target)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.dqn.state_dict(), path)

    def load(self, path):
        self.dqn.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._target_hard_update()
