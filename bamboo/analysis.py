import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict
import numpy as np

def bar_plot_avg_metrics(data: list[dict]):
    """
    The function plots two bar plots - one for average time taken and one for average score for each parameter value.
    The plots are sorted in ascending order of average values. Each parameter is represented by a different color.
    
    Parameters
    ----------
    data : list of dict
        Each dictionary in the list represents a configuration and contains 'Params', 'Time Taken', and 'Score' keys.
        'Params' is a dictionary itself and contains the parameter values.
    """
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
    unique_param_values = {param: sort_with_none(list(set(param_values[param]))) for param in parameters}
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


def plot_swarm(configurations: list[dict], parameter: str, color: str = 'purple'):
    """
    Plots a swarm plot for the scores of different configurations of a given parameter.

    Parameters
    ----------
    configurations : list of dict
        Each dictionary in the list represents a configuration and contains 'Params' and 'Score' keys. 
        'Params' is a dictionary itself and contains the parameter values.
    parameter : str
        The parameter for which the swarm plot will be generated.
    color : str, optional
        The color of the points in the swarm plot. Default is 'purple'.
    """
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


def plot_precision_recall_curve(data_per_class: list[dict], classes: list):
    """
    Plots the Precision-Recall curve for multiple classes.

    Parameters
    ----------
    data_per_class : list of dict
        Each dictionary contains 'precision' and 'recall' data for a specific class.
    classes : list
        A list of the classes to be plotted.
    """
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


def plot_roc_curve(data_per_class: list[dict], classes: list):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for multiple classes.

    Parameters
    ----------
    data_per_class : list of dict
        Each dictionary contains 'fpr' (False Positive Rate), 'tpr' (True Positive Rate), and 'roc_auc' (ROC AUC) data for a specific class.
    classes : list
        A list of the classes to be plotted.
    """
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


def plot_det_curve(data_per_class: list[dict], classes: list):
    """
    Plots the Detection Error Tradeoff (DET) curve for multiple classes.

    Parameters
    ----------
    data_per_class : list of dict
        Each dictionary contains 'far' (False Acceptance Rate) and 'fnr' (False Non-Acceptance Rate) data for a specific class.
    classes : list
        A list of the classes to be plotted.
    """
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
    """
    This function plots multiple graphs side by side on a single figure.

    Parameters
    ----------
    *plot_funcs : functions
        Multiple plotting function calls. Each function should draw a plot when called.

    Optional parameters
    -------------------
    titles : list of str 
        A list of titles for each subplot. Default is None.
    num_cols : int 
        Number of columns to use for arranging subplots. Default is 2.
    figsize : tuple 
        Figure size as a tuple of width and height in inches. Default is (15, 5).

    Note
    ----
    The plotting functions specified in plot_funcs must not call plt.show(), as this function 
    will call it after arranging all subplots.
    """
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


def sort_with_none(lst: list):
    """
    Sorts a list that can contain a mix of None, string and other comparable types.
    """
    return sorted(lst, key=lambda x: (x is None, isinstance(x, str), x))
