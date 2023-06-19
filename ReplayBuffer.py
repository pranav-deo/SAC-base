import numpy as np
import torch

def np2tensor(tuple_arr, device):
    return [torch.tensor(arr, dtype=torch.float32, device=device) for arr in tuple_arr]

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim, device, seed) -> None:
        self._obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._act = np.zeros((max_size, act_dim), dtype=np.float32)
        self._rew = np.zeros((max_size, 1), dtype=np.float32)
        self._not_dones = np.zeros((max_size, 1), dtype=np.float32)

        self.size = 0
        self._max_size = max_size
        self._ptr = 0
        self._device = device

        self._rng = np.random.Generator(np.random.PCG64(seed=seed))

    def add_step(self, obs, next_obs, act, rew, nd):
        self._obs[self._ptr] = np.array(obs, dtype=np.float32)
        self._next_obs[self._ptr] = np.array(next_obs, dtype=np.float32)
        self._act[self._ptr] = np.array(act, dtype=np.float32)
        self._rew[self._ptr] = np.array(rew, dtype=np.float32)
        self._not_dones[self._ptr] = np.array(nd, dtype=np.float32)
        self._ptr = (self._ptr + 1) % self._max_size
        self.size = min(self.size + 1 , self._max_size)

    def sample(self, batch_size):
        idx = self._rng.choice(self.size, batch_size, replace=False)
        return np2tensor((self._obs[idx], self._next_obs[idx], self._act[idx],
                self._rew[idx], self._not_dones[idx]), self._device)