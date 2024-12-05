from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluation_graphs(history, model, X_test, y_test, label_encoder):
    # Plotting training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Predict the values from the validation dataset
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions classes to one hot vectors 

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, classes=label_encoder.classes_)
    
    plt.show()

# Function to compare accuracies of three models
def compare_accuracies(hist_rnn, hist_dnn, hist_bilstm):
    plt.figure(figsize=(10, 6))
    plt.plot(hist_rnn.history['val_accuracy'], label='RNN Val Accuracy')
    plt.plot(hist_dnn.history['val_accuracy'], label='DNN Val Accuracy')
    plt.plot(hist_bilstm.history['val_accuracy'], label='BiLSTM Val Accuracy')
    plt.title('Model Comparison - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def print_evaluation_scores(y_true, y_pred,classes):
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Calculate averages for precision, recall, and F1-score
    
    precision_avg = report['macro avg']['precision']
    recall_avg = report['macro avg']['recall']
    f1_score_avg = report['macro avg']['f1-score']
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(f"Average Precision: {precision_avg:.2f}")
    print(f"Average Recall: {recall_avg:.2f}")
    print(f"Average F1-Score: {f1_score_avg:.2f}")

