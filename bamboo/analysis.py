import matplotlib.pyplot as plt

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
    plt.show()

def plot_roc_curve(data_per_class, classes):
    # Define colors
    colors = ['teal', 'purple', 'darkorange', 'green']

    # For each class
    for i, color in zip(range(len(classes)), colors):
        # Get data for this class
        data = data_per_class[i]

        # Plot the ROC curve
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, data['roc_auc']))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves')
    plt.legend(loc="lower right")
    plt.show()




