import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "logs/training_analytics.csv"
ROLLING_WINDOW = 20

def plot_metric(df, column_name, title, ylabel):
    plt.figure(figsize=(10, 5))

    plt.plot(
        df["episode"],
        df[column_name],
        alpha=0.35,
        label=f"Raw {column_name}"
    )

    rolling = df[column_name].rolling(ROLLING_WINDOW).mean()

    plt.plot(
        df["episode"],
        rolling,
        linewidth=2,
        label=f"{ROLLING_WINDOW}-Episode Rolling Average"
    )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_q_values(df):
    plt.figure(figsize=(10, 5))

    plt.plot(
        df["episode"],
        df["mean_max_q"].rolling(ROLLING_WINDOW).mean(),
        label="Mean Max Q"
    )

    plt.plot(
        df["episode"],
        df["mean_q"].rolling(ROLLING_WINDOW).mean(),
        label="Mean Q"
    )

    plt.title("Q-Value Trends")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(CSV_PATH)

    plot_metric(
        df,
        "episode_reward",
        "Episode Reward vs Episode",
        "Reward"
    )

    plot_metric(
        df,
        "episode_steps",
        "Episode Steps vs Episode",
        "Steps"
    )

    plot_metric(
        df,
        "average_loss",
        "Average Loss vs Episode",
        "Loss"
    )

    plot_q_values(df)

    plot_metric(
        df,
        "mean_q_gap",
        "Mean Q Gap vs Episode",
        "Q Gap"
    )


if __name__ == "__main__":
    main()