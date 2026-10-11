"""Small shared temporal encoder over complete preaction timetable events."""

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.transit.native_timetable_context import event_features

WIDTH, KERNEL, OUTPUT_DIM = 8, 3, 16


def training_scales(rows):
    events = np.concatenate([event_features(r).reshape(-1, 6) for r in rows])
    center, scale = events.mean(0), events.std(0)
    scale[scale < 1e-8] = 1.
    return center, scale


def batch_events(rows, center, scale):
    arrays = [event_features(r) for r in rows]
    length = max(a.shape[1] for a in arrays)
    events = np.zeros((len(rows)*6, length, 6), dtype=np.float32)
    mask = np.zeros((len(rows)*6, length, 1), dtype=np.float32)
    for i, a in enumerate(arrays):
        events[i*6:(i+1)*6, :a.shape[1]] = (a-center)/scale
        mask[i*6:(i+1)*6, :a.shape[1]] = 1.
    return torch.from_numpy(events), torch.from_numpy(mask)


class TimetableEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(6, WIDTH)
        self.temporal = nn.Conv1d(WIDTH, WIDTH, KERNEL, padding=1)

    def forward(self, events, mask):
        hidden = torch.tanh(self.input(events))*mask
        hidden = torch.tanh(self.temporal(hidden.transpose(1, 2)).transpose(1, 2))
        mean = (hidden*mask).sum(1)/mask.sum(1)
        maximum = hidden.masked_fill(mask == 0, -torch.inf).amax(1)
        return torch.cat((mean, maximum), dim=-1)


def encode_numpy(row, model):
    events = (event_features(row)-model["center"])/model["scale"]
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    hidden = np.tanh(events@w["input.weight"].T+w["input.bias"])
    padded = np.pad(hidden, ((0, 0), (1, 1), (0, 0)))
    convolved = np.broadcast_to(w["temporal.bias"], hidden.shape).copy()
    for offset in range(KERNEL):
        convolved += padded[:, offset:offset+hidden.shape[1]]@w["temporal.weight"][:, :, offset].T
    hidden = np.tanh(convolved)
    return np.c_[hidden.mean(1), hidden.max(1)]
