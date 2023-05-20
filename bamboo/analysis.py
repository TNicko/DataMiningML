import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib import cycler
from collections import defaultdict
import numpy as np

def bar_plot_avg_metrics(data):
    # Prepare data
    parameters = [param for param in data[0]['Params']]
    param_values = {param: [] for param in parameters}
    time_values = {param: [] for param in parameters}
    score_values = {param: [] for param in parameters}

    cmap = plt.cm.coolwarm
    colors = defaultdict(str)
    for i, param in enumerate(parameters):
        colors[param] = cmap(i / len(parameters) * 1.5)
    legend_elements = [Line2D([0], [0], color=colors[param], lw=4, label=param) for param in parameters]

    for item in data:
        for param in parameters:
            param_values[param].append(item['Params'][param])
            time_values[param].append(item['Time Taken'])
            score_values[param].append(item['Score'])

    # Calculate averages
    avg_time = {param: defaultdict(float) for param in parameters}
    avg_score = {param: defaultdict(float) for param in parameters}
    count_metrics = {param: defaultdict(int) for param in parameters}
    for param in parameters:
        for value, time, score in zip(param_values[param], time_values[param], score_values[param]):
            avg_time[param][value] += time
            avg_score[param][value] += score
            count_metrics[param][value] += 1
    avg_time = {
        param: {value: avg_time[param][value] / count_metrics[param][value]
                for value in avg_time[param]} for param in parameters}
    avg_score = {
        param: {value: avg_score[param][value] / count_metrics[param][value]
                for value in avg_score[param]} for param in parameters}

    # Prepare for plotting
    unique_param_values = {param: sorted(list(set(param_values[param]))) for param in parameters}
    bar_width = 1 / (len(unique_param_values) + 1)  # leave space between groups

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    sorted_times = []
    sorted_scores = []
    for param in parameters:
        for value in unique_param_values[param]:  
            sorted_times.append((avg_time[param][value], value, param))
            sorted_scores.append((avg_score[param][value], value, param))
    sorted_times.sort(key=lambda x: x[0])
    sorted_scores.sort(key=lambda x: x[0])

    # Plotting Time Taken
    x_position = 0
    for item in sorted_times:
        x_position += 0.3
        axs[0].bar(x_position, item[0], width=bar_width, color=colors[item[2]], label='Time Taken')
    axs[0].set_xticks(np.arange(len(sorted_times)) * 0.3 + 0.3)
    axs[0].set_xticklabels([value[1] for value in sorted_times], rotation=40)
    axs[0].set_ylabel('Average Time Taken')
    axs[0].set_title('Average Time Taken for Each Parameter Value')
    axs[0].legend(handles=legend_elements, loc='upper left')

        # Plotting Score
    x_position = 0
    for item in sorted_scores:
        x_position += 0.3
        axs[1].bar(x_position, item[0], width=bar_width, color=colors[item[2]], label='Score')
    axs[1].set_xticks(np.arange(len(sorted_scores)) * 0.3 + 0.3)
    axs[1].set_xticklabels([value[1] for value in sorted_scores], rotation=40)
    axs[1].set_ylabel('Average Score')
    axs[1].set_title('Average Score for Each Parameter Value')
    axs[1].legend(handles=legend_elements, loc='upper left')

    plt.show()

# def bar_plot_avg_metric(data, metric: str ='time'):
#     # Validate input
#     assert metric in ['time', 'score'], 'Invalid metric'

#     if metric == 'time':
#         metric = 'Time Taken'
#     if metric == 'score':
#         metric = 'Score'

#     # Prepare data
#     parameters = [param for param in data[0]['Params']]
#     param_values = {param: [] for param in parameters}
#     metric_values = {param: [] for param in parameters}

#     cmap = plt.cm.coolwarm
#     colors = defaultdict(str)
#     for i, param in enumerate(parameters):
#         colors[param] = cmap(i / len(parameters) * 1.5)
#     legend_elements = [Line2D([0], [0], color=colors[param], lw=4, label=param) for param in parameters]

#     for item in data:
#         for param in parameters:
#             param_values[param].append(item['Params'][param])
#             metric_values[param].append(item[metric])

#     # Calculate averages
#     avg_metrics = {param: defaultdict(float) for param in parameters}
#     count_metrics = {param: defaultdict(int) for param in parameters}
#     for param in parameters:
#         for value, met_value in zip(param_values[param], metric_values[param]):
#             avg_metrics[param][value] += met_value
#             count_metrics[param][value] += 1
#     avg_metrics = {
#         param: {value: avg_metrics[param][value] / count_metrics[param][value]
#                 for value in avg_metrics[param]} for param in parameters}

#     # Prepare for plotting
#     unique_param_values = {param: sorted(list(set(param_values[param]))) for param in parameters}
#     bar_width = 1 / (len(unique_param_values) + 1)  # leave space between groups

#     fig, ax = plt.subplots()

#     sorted_metrics = []
#     for param in parameters:
#         for value in unique_param_values[param]:  
#             sorted_metrics.append((avg_metrics[param][value], value, param))
#     sorted_metrics.sort(key=lambda x: x[0])

#     x_position = 0
#     for item in sorted_metrics:
#         x_position += 0.3
#         ax.bar(x_position, item[0], width=bar_width, color=colors[item[2]])

#     # Labeling
#     ax.set_xticks(np.arange(len(sorted_metrics)) * 0.3 + 0.3)
#     ax.set_xticklabels([value[1] for value in sorted_metrics], rotation=40)
#     plt.title(f'Average {metric} for Each Parameter Value')
#     plt.xlabel('Parameter Value')
#     plt.ylabel(f'Average {metric}')
#     ax.legend(handles=legend_elements, loc='upper left')

def plot_swarm(configurations, parameter: str, color: str = 'purple'):
    # Extract the parameter values and scores
    params = [config['Params'][parameter] for config in configurations]
    scores = [config['Score'] for config in configurations]

    # Plot the swarm plot
    sns.swarmplot(x=params, y=scores, color=color)

    # Set the axis labels and title
    plt.xlabel(parameter)
    plt.ylabel('Score')
    plt.title('Scores for Different Configurations')

    # Show the plot
    plt.show()

def plot_precision_recall_curve(data_per_class, classes):
    colors = ['teal', 'purple', 'darkorange', 'green']

    # For each class
    for i in range(len(classes)):
        # Get data for this class
        data = data_per_class[i]

        # Plot the precision-recall curve for the class
        plt.plot(data['recall'], data['precision'], color=colors[i], lw=2, label='class {}'.format(i))
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")

def plot_roc_curve(data_per_class, classes):
    colors = ['teal', 'purple', 'darkorange', 'green']

    # For each class
    for i, color in zip(range(len(classes)), colors):
        # Get data for this class
        data = data_per_class[i]

        # Plot the ROC curve
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2, label='Class {0} (area = {1:0.2f})'.format(i, data['roc_auc']))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves')
    plt.legend(loc="lower right")


def plot_det_curve(data_per_class, classes):
    colors = ['teal', 'purple', 'darkorange', 'green']

    # For each class
    for i, color in zip(range(len(classes)), colors):
        # Get data for this class
        data = data_per_class[i]

        # Plot the DET curve
        plt.plot(data['far'], data['fnr'], color=color, lw=2, label='Class {0}'.format(i))

    # Set plot properties
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.legend(loc="upper right")
    plt.grid(True)


def plot_side_by_side(*plot_funcs, titles=None, num_cols=2, figsize=(15, 5)):
    num_plots = len(plot_funcs)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.ravel()

    for i, plot_func in enumerate(plot_funcs):
        plt.sca(axes[i])
        plot_func()
        if titles:
            plt.title(titles[i])

    # Hide empty subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
