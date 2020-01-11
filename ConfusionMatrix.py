from CommonVariables import *
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Function for plotting the confusion matrix.
# taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.htm
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          name='Confusion Matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_folder+'/Data/'+name+'.png')


# Compute the confusion matrix.
print('Confusion Matrix')
ground_truth = np.load(output_folder + '/Data/Ground_truth.npy')
predictions = np.load(output_folder + '/Data/Predictions.npy')
cm = confusion_matrix(ground_truth, predictions)
classes_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutrality', 'Sadness', 'Surprise']
title = 'Confusion Matrix'

# Plot the non-normalized confusion matrix.
plt.figure()
plot_confusion_matrix(cm, classes_labels, title='Confusion Matrix', name='Confusion Matrix')
# Plot the normalized confusion matrix.
plt.figure()
plot_confusion_matrix(cm, classes_labels, title='Normalized Confusion Matrix', normalize=True, name='Normalized Confusion Matrix')
plt.show()