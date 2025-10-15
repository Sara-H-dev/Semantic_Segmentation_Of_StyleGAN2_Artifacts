import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lr_range_test_file = 'lr_range_test.csv'

csv_reader = pd.read_csv('./lr_range_test.csv')
print(csv_reader.head())

csv_reader["smoothed_train_loss"] = csv_reader["train_loss"].ewm(span=20, adjust=False).mean()
csv_reader["smoothed_val_loss"] = csv_reader["val_loss"].ewm(span=20, adjust=False).mean()
plt.figure(figsize=(8, 6))
plt.plot(csv_reader["lr"], csv_reader["smoothed_train_loss"], label="Smoothed Train Loss", linewidth=2)
#plt.plot(csv_reader["lr"], csv_reader["train_loss"], color='lightblue', alpha=0.3, label="Raw Train Loss")
plt.plot(csv_reader["lr"], csv_reader["smoothed_val_loss"], color='red', label="Smoothed Validation Loss", linewidth=2)
#plt.plot(csv_reader["lr"], csv_reader["val_loss"], color='salmon', alpha=0.3, label="Raw Validation Loss")
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.ylim(0, 2) 
plt.legend(loc="best")
plt.title("Learning Rate Range Test")
plt.grid(True)
plt.savefig("weight_decay_test.png", dpi=300) 
plt.show()