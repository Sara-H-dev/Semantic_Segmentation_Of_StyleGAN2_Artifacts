import matplotlib.pyplot as plt
import pandas as pd


csv_reader = pd.read_csv('./val_metric_all_epoch.csv')
csv_reader_real = pd.read_csv('./val_metric_real_epoch.csv')
csv_reader_fake = pd.read_csv('./val_metric_fake_epoch.csv')
print(csv_reader.head())

plt.figure(figsize=(8, 6))
plt.plot(csv_reader["epoch"], csv_reader["mean_val_loss"], label="Validation Loss", linewidth=2)
plt.plot(csv_reader_real["epoch"], csv_reader_real["mean_val_loss_real"], label="Validation Loss Real", linewidth=2)
plt.plot(csv_reader_fake["epoch"], csv_reader_fake["mean_val_loss_fake"], label="Validation Loss Fake", linewidth=2)
#plt.plot(csv_reader["lr"], csv_reader["train_loss"], color='lightblue', alpha=0.3, label="Raw Train Loss")
plt.plot(csv_reader["epoch"], csv_reader["mean_train_loss"], color='red', label="Train Loss", linewidth=2)
#plt.plot(csv_reader["lr"], csv_reader["val_loss"], color='salmon', alpha=0.3, label="Raw Validation Loss")
#plt.xscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1) 
plt.legend(loc="best")
plt.title("Loss Function Plot")
plt.grid(True)
plt.savefig("loss_plot_per_epoch.png", dpi=300) 
plt.show()