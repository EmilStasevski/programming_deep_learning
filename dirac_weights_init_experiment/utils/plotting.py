import matplotlib.pyplot as plt


def plot_experiment_results(results_dict, title):
    plt.figure(figsize=(10, 6))
    for label, losses in results_dict.items():
        plt.plot(losses, label=label, alpha=0.8)

    plt.title(title)
    plt.xlabel('Training Iterations (x100 batches)')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_plot.png')
    print("Plot saved as 'experiment_plot.png'")