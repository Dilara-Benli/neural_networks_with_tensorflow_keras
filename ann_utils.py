import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from metric_functions import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc

class ModelTraining:
    def __init__(self, dataset_path):
        self.dataset = pd.read_excel(dataset_path)
        self.dataset = self.dataset.drop("ID", axis=1) # benim veri setime özel

        self.features = self.dataset.iloc[:, :-1].values 
        self.labels = self.dataset.iloc[:, -1].values     # son sütun = etiket, olarak kabul edilir
        self.labels = tf.keras.utils.to_categorical(self.labels)  

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.labels, test_size=0.3, random_state=10)

        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    # Yapay Sinir Ağı (Artificial Neural Network), MLP: Multi-Layer Perceptron
    def create_ann_model(self):
        ann_model = tf.keras.models.Sequential()
        ann_model.add(tf.keras.layers.Dense(10, activation="relu", input_shape=(self.x_train.shape[1],))) # 30 features
        ann_model.add(tf.keras.layers.Dense(10, activation="relu"))
        ann_model.add(tf.keras.layers.Dense(10, activation="relu"))
        ann_model.add(tf.keras.layers.Dense(10, activation="relu"))
        ann_model.add(tf.keras.layers.Dense(10, activation="relu"))
        ann_model.add(tf.keras.layers.Dense(2, activation="softmax")) # 2 class

        ann_model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                                     metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),  
                                              specificity, F1_Score, tf.keras.metrics.AUC(), cohen_kappa])
        
        history = ann_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                                           epochs=20, batch_size=16)
        
        return ann_model, history
    
    def save_model_and_history(self, model, history, model_path, history_path):
        model.save(model_path)
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        print(f"Model saved to {model_path} and history saved to {history_path}")

    def load_model_and_history(self, model_path, history_path):
        model = tf.keras.models.load_model(model_path, custom_objects={
            'specificity': specificity,
            'F1_Score': F1_Score,
            'cohen_kappa': cohen_kappa
        })
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Model loaded from {model_path} and history loaded from {history_path}")
        return model, history
    
    def calculate_evaluation_metrics(self, model):
        pred_prob = model.predict(self.x_test)
        predictions = tf.argmax(pred_prob, axis=1).numpy()
        true_labels = tf.argmax(self.y_test, axis=1).numpy()

        conf_matrix = confusion_matrix(true_labels, predictions)

        accuracy = accuracy_score(true_labels, predictions) # doğruluk
        precision = precision_score(true_labels, predictions) # kesinlik
        recall = recall_score(true_labels, predictions) # duyarlılık
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) # özgüllük
        f1 = f1_score(true_labels, predictions)
        auc_score = roc_auc_score(true_labels, predictions)
        kappa = cohen_kappa_score(true_labels, predictions)

        print(f"Accuracy Score: {accuracy:.3f}\nPrecision Score: {precision:.3f}\nRecall Score: {recall:.3f}\nSpecificity Score: {specificity:.3f}\nF1 Score: {f1:.3f}\nAuc Score: {auc_score:.3f}\nKappa Score: {kappa:.3f}")

        return accuracy, precision, recall, specificity, f1, auc_score, kappa
    
    def plot_evaluation_metrics(self, history, metrics):
        num_metrics = len(metrics)
        num_cols = 4  # sütun
        num_rows = (num_metrics + num_cols - 1) // num_cols  # satır

        plt.figure(figsize=(12, 3 * num_rows))

        for i, metric in enumerate(metrics):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(history[metric], label='Train ' + metric.capitalize()) 
            plt.plot(history[f'val_{metric}'], label='Val ' + metric.capitalize())
            #plt.title(f'Train and Val {metric.capitalize()}') # Train and Validation
            plt.title(metric)
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_conf_matrix(self, model):
        pred_prob = model.predict(self.x_test)
        predictions = tf.argmax(pred_prob, axis=1).numpy()
        true_labels = tf.argmax(self.y_test, axis=1).numpy()

        conf_matrix = confusion_matrix(true_labels, predictions)
        conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Benign", "Malignant"])
        conf_matrix_disp.plot()
    
    def plot_roc_curve(self, model):
        pred_prob = model.predict(self.x_test)
        fpr, tpr, _ = roc_curve(self.y_test[:, 1], pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

