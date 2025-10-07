import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history, title="History Training", figsize=(18, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["valid_loss"], label="Valid loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train acc")
    axes[1].plot(history["valid_acc"], label="Valid acc")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Acc")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", normalize=False, figsize=(10, 8)):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Pradicted Label")
    plt.tight_layout()
    plt.show()

def plot_loss_comparision(histories, labels, title="Loss Comparision"):
    plt.figure(figsize=(12, 8))

    for history, label in zip(histories, labels):
        plt.plot(history["train_loss"], label=f"{label} - Train", linestyle='-')
        plt.plot(history["valid_loss"], label=f"{label} - Valid", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_fps(metrics, title="Accuracy vs FPS", figsize=(12, 6), range_acc=(0.9, 1.0), range_fps=(2000, 7000)):
    models = list(metrics.keys())
    accuracies = [metrics[model]["acc"] for model in models]
    fps_values = [metrics[model]["fps"] for model in models]

    fig, ax1 = plt.subplots(figsize=figsize)

    bars = ax1.bar(models, accuracies, color="#55A868", alpha=0.8, width=0.6)
    ax1.set_ylabel("Accuracy", fontweight="bold")
    ax1.set_ylim(range_acc)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    line = ax2.plot(models, fps_values, color="#DD8452", marker="o",
                    linewidth=2, markersize=6, label="FPS")
    ax2.set_ylabel("FPS", fontweight="bold")
    ax2.set_ylim(range_fps)

    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.show()

def plot_param(metrics, title="Model parameter size", figsize=(12, 8)):
    models = list(metrics.keys())
    sizes_mb = [metrics[model]["size_mb"] for model in models]

    sorted_data = sorted(zip(models, sizes_mb), key=lambda x: x[1])
    models_sorted, sizes_sorted = zip(*sorted_data)

    plt.figure(figsize=figsize)

    bars = plt.barh(models_sorted, sizes_sorted, color='#4472C4', alpha=0.8)

    for i, (bar, size) in enumerate(zip(bars, sizes_sorted)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{size:.1f} MB', va='center', ha='left', fontweight='bold')
    
    plt.xlabel('Model size (MB)', fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14, pad=20)
    plt.xlim(0, max(sizes_sorted) * 1.15)

    plt.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

