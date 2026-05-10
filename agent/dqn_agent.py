from agent.dqn_network import DQNNetwork
import random
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, env, device, lr = 1e-4, gamma = 0.99, epsilon_start = 1.0, epsilon_end = 0.08,
                 epsilon_decay = 3500000):
        self.env = env
        self.device = device
        self.num_actions = env.num_actions
        self.gamma = gamma
        self.steps_done = 0
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.online_net = DQNNetwork(env).to(device)
        self.target_net = DQNNetwork(env).to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr = lr)

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state, dtype = torch.float32, device = self.device) / 255.0
                state = state.unsqueeze(0)

                q_values = self.online_net(state)
                action_idx = torch.argmax(q_values, dim = 1).item()
        self.epsilon = max(self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay,
                           self.epsilon_end)
        self.steps_done += 1
        return action_idx

    def select_greedy_action(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype = torch.float32, device = self.device) / 255.0
            state = state.unsqueeze(0)

            q_values = self.online_net(state)
            action_idx = torch.argmax(q_values, dim = 1).item()

        return action_idx

    def get_q_stats(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) / 255.0
            state = state.unsqueeze(0)

            q_values = self.online_net(state).squeeze(0)
            max_q = q_values.max().item()
            mean_q = q_values.mean().item()

            top2 = torch.topk(q_values, k = 2).values
            q_gap = (top2[0] - top2[1]).item()

            return max_q, mean_q, q_gap

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)
        q_values = self.online_net(states)
        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim = 1).values
            target_q_values = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        loss = F.smooth_l1_loss(action_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, total_steps = None, episode = None):
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "gamma": self.gamma,
            "total_steps": total_steps,
            "episode": episode
        }
        torch.save(checkpoint, path)

    def load(self, path, epsilon_override=None):
        checkpoint = torch.load(path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint["epsilon"]

        if epsilon_override is not None:
            self.epsilon = epsilon_override

        self.steps_done = checkpoint["steps_done"]
        self.gamma = checkpoint["gamma"]

        self.target_net.eval()

        return {
            "total_steps": checkpoint.get("total_steps", 0),
            "episode": checkpoint.get("episode", 0),
        }

