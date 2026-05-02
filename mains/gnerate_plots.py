import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 1. Plotting the Training/Validation Loss
metrics = pd.read_csv("logs/exp2_metrics.csv")
plt.figure(figsize=(10, 5))
plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
plt.title("Experiment 2: Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.show()

# 2. Plotting the Confusion Matrix
preds_df = pd.read_csv("logs/exp2_test_predictions.csv")
cm = confusion_matrix(preds_df['True_Label'], preds_df['Predicted_Label'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(preds_df['True_Label'].unique()), 
            yticklabels=sorted(preds_df['True_Label'].unique()))
plt.title("Experiment 2: Confusion Matrix")
plt.xlabel("Predicted Gesture")
plt.ylabel("Actual Gesture")
plt.show()

# 3. Print Precision, Recall, F1-Score
print(classification_report(preds_df['True_Label'], preds_df['Predicted_Label']))