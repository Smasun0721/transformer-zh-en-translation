import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止图表乱码
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training_metrics(csv_file="training_log.csv"):
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件 {csv_file}")
        return

    df = pd.read_csv(csv_file)
    # 清理可能存在的空行或格式错乱行
    df = df.dropna(subset=["Type"])
    df = df.sort_values(by="Global_Step")

    df_train = df[df["Type"] == "Train"]
    df_val = df[df["Type"] == "Val"]

    if df_train.empty:
        print("⚠️ 还没有训练数据")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Transformer 训练全景分析", fontsize=18, fontweight="bold")

    # 1. 损失 (Loss)
    axs[0, 0].plot(
        df_train["Global_Step"],
        df_train["Loss"],
        label="Train Loss",
        color="#1f77b4",
        alpha=0.6,
        linewidth=1,
    )
    if not df_val.empty:
        axs[0, 0].plot(
            df_val["Global_Step"],
            df_val["Loss"],
            label="Val Loss",
            color="#ff7f0e",
            linewidth=2.5,  # 加粗一点，没有 marker 也清晰
        )
    axs[0, 0].set_title("损失 (Loss)", fontsize=14)
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()

    # 2. 准确率 & 困惑度
    ax_twin = axs[0, 1].twinx()
    axs[0, 1].plot(
        df_train["Global_Step"],
        df_train["Acc"],
        label="Train Acc",
        color="#2ca02c",
        alpha=0.6,
        linewidth=1,
    )
    if not df_val.empty:
        axs[0, 1].plot(
            df_val["Global_Step"],
            df_val["Acc"],
            label="Val Acc",
            color="#d62728",
            linewidth=2.5,
        )
        ax_twin.plot(
            df_val["Global_Step"],
            df_val["PPL"],
            label="Val PPL",
            color="#9467bd",
            linestyle="-.",
            linewidth=2.5,
        )

    axs[0, 1].set_title("准确率 (Acc) 与 困惑度 (PPL)", fontsize=14)
    axs[0, 1].set_ylabel("Accuracy")
    ax_twin.set_ylabel("PPL")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    lines_1, labels_1 = axs[0, 1].get_legend_handles_labels()
    lines_2, labels_2 = ax_twin.get_legend_handles_labels()
    axs[0, 1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    # 3. 学习率 (LR)
    axs[1, 0].plot(
        df_train["Global_Step"],
        df_train["LR"],
        label="Learning Rate",
        color="#8c564b",
        linewidth=2,
    )
    axs[1, 0].set_title("学习率 (Learning Rate)", fontsize=14)
    axs[1, 0].set_xlabel("Global Steps")
    axs[1, 0].set_ylabel("LR")
    axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()

    # 4. 梯度范数 (Gradient Norm)
    axs[1, 1].plot(
        df_train["Global_Step"],
        df_train["GradNorm"],
        label="Gradient Norm",
        color="#e377c2",
        alpha=0.7,
        linewidth=1,
    )
    axs[1, 1].set_title("梯度范数 (监控是否爆炸)", fontsize=14)
    axs[1, 1].set_xlabel("Global Steps")
    axs[1, 1].set_ylabel("Norm")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("training_overview.png", dpi=300)
    print("✅ 图表已保存为 training_overview.png")
    if os.environ.get("SHOW_PLOTS") == "1":
        plt.show()


if __name__ == "__main__":
    plot_training_metrics()