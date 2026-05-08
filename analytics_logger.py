import csv
import os

class EpisodeAnalytics:
    def __init__(self, path):
        self.path = path
        self.fieldnames = [
            "episode",
            "total_steps",
            "episode_steps",
            "episode_reward",
            "episode_env_reward",
            "lives_lost",
            "epsilon",
            "average_loss",
            "mean_max_q",
            "mean_q",
            "mean_q_gap"
        ]

        os.makedirs(os.path.dirname(path), exist_ok = True)

        if not os.path.exists(path):
            with open(self.path, "w", newline = "") as f:
                writer = csv.DictWriter(f, fieldnames = self.fieldnames)
                writer.writeheader()

    def write_episode(self, row):
        with open(self.path, "a", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = self.fieldnames)
            writer.writerow(row)