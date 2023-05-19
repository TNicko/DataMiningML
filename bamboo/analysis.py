from sklearn.calibration import label_binarize
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, X_test, y_test, classes):

    colors = ['teal', 'purple', 'darkorange', 'green']

    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)

    # Compute the prediction probabilities
    y_scores = model.predict_proba(X_test)

    # For each class
    for i in range(len(classes)):
        # Compute the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        
        # Plot the precision-recall curve for the class
        plt.plot(recall, precision, color=colors[i], lw=2, label='class {}'.format(i))
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()






