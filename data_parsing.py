import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('compiled_data/mismatch.csv')

y_true = df['actual']
y_pred = df['predicted']

# Combine unique labels from both actual and predicted
all_labels = np.unique(np.concatenate((y_true.unique(), y_pred.unique())))

# Calculate confusion matrix with explicit labels
cm = confusion_matrix(y_true, y_pred, labels=all_labels)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()
