import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, cell_size=64):
        super(ActorCriticLSTM, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.cell_size = cell_size

        self.lstm = nn.LSTM(obs_space, cell_size)
        self.actor = nn.Linear(cell_size, num_outputs)
        self.critic = nn.Linear(cell_size, 1)

        self.hidden_state = torch.zeros(1, 1, cell_size)
        self.cell_state = torch.zeros(1, 1, cell_size)

    def forward(self, x):
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(x.view(1, 1, -1), (self.hidden_state, self.cell_state))
        actor_out = F.softmax(self.actor(lstm_out.view(1, -1)), dim=-1)
        critic_out = self.critic(lstm_out.view(1, -1))
        return actor_out, critic_out

    def reset_states(self):
        self.hidden_state.zero_()
        self.cell_state.zero_()
