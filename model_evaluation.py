import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Split the data into training and testing sets (not needed for cross-validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes model
def naive_bayes_model(X, y):
    nb = GaussianNB()
    nb.fit(X, y)
    return nb

# Random Forest model
def random_forest_model(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

# Confusion matrix
def compute_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    cm= confusion_matrix(y, y_pred)
    return cm

# Compute error rates using 10-fold cross-validation and t-test
def compare_models(model1, model2, X, y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores1 = cross_val_score(model1, X, y, cv=kfold)
    scores2 = cross_val_score(model2, X, y, cv=kfold)
    t, p = ttest_rel(scores1, scores2)
    return scores1.mean(), scores2.mean(), t, p

# Draw ROC curve for both models
def draw_roc_curve(model1, model2, X, y):
    fpr1, tpr1, _ = roc_curve(y, model1.predict_proba(X)[:, 1])
    fpr2, tpr2, _ = roc_curve(y, model2.predict_proba(X)[:, 1])
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    plt.figure()
    lw = 2
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue',
             lw=lw, label='Random Forest (AUC = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

# Load the breast cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train the Naive Bayes and Random Forest models
nb_model = naive_bayes_model(X, y)
rf_model = random_forest_model(X, y)

# Compute confusion matrices
nb_cm = compute_confusion_matrix(nb_model, X, y)
rf_cm = compute_confusion_matrix(rf_model, X, y)

def plot_cm(cm, class_labels):
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

plot_cm(nb_cm, ['Negative', 'Positive'])

plot_cm(rf_cm, ['Negative', 'Positive'])

# Compare error rates using 10-fold cross-validation and t-test
nb_score, rf_score, t, p = compare_models(nb_model, rf_model, X, y)
print('Naive Bayes score: {:.3f}'.format(nb_score))
print('Random Forest score: {:.3f}'.format(rf_score))
print('t-statistic: {:.3f}'.format(t))
print('p-value: {:.3f}'.format(p))


# Draw ROC curve for both models
draw_roc_curve(nb_model, rf_model, X, y)
plt.legend(loc="lower right")
plt.show()