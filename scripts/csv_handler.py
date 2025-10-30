import os
import csv

class CSV_Handler:
    def __init__(self, log_save_path):
        os.makedirs(log_save_path, exist_ok=True)
         # lr_range_test
        lr_range_test_file = os.path.join(log_save_path, "lr_range_test.csv")
        csv_file = open(lr_range_test_file, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "lr", "train_loss", "val_loss"])

        batch_test_file = os.path.join(log_save_path, "batch_test_file.csv")
        csv_file_batch = open(batch_test_file, "w", newline="")
        csv_writer_batch = csv.writer(csv_file_batch)
        csv_writer_batch.writerow(["epoch", "case_name"])
        
        # val metric per epoch for all
        val_metric_file = os.path.join(log_save_path, "val_metric_all_epoch.csv")
        csv_all_epoch_file = open(val_metric_file, "w", newline="")
        csv_all_epoch = csv.writer(csv_all_epoch_file)
        csv_all_epoch.writerow(["epoch","mean_accuracy","mean_val_loss", "mean_train_loss","mean_confusion_matrix_bin", "mean_confusion_matrix_soft"])
        
        # val metric per epoch for real images
        val_metric_file = os.path.join(log_save_path, "val_metric_real_epoch.csv")
        csv_real_epoch_file = open(val_metric_file, "w", newline="")
        csv_real_epoch = csv.writer(csv_real_epoch_file)
        csv_real_epoch.writerow(["epoch","accuracy_real","mean_confusion_matrix_bin", "mean_confusion_matrix_soft","mean_val_loss_real"])
        
        # val metric per epoch for fake images
        val_metric_file = os.path.join(log_save_path, "val_metric_fake_epoch.csv")
        csv_fake_epoch_file = open(val_metric_file, "w", newline="")
        csv_fake_epoch = csv.writer(csv_fake_epoch_file)
        csv_fake_epoch.writerow(["epoch","mean_accuracy","mean_val_loss_fake","mean_confusion_matrix_bin","mean_confusion_matrix_soft",
                                "mean_bin_accuracy","mean_bin_recall","mean_bin_precision","mean_bin_IoU","mean_bin_dice","mean_bin_f1","mean_i_soft_dice","mean_i_soft_iou"])
        
        # val metric per batch real
        val_metric_file = os.path.join(log_save_path, "val_metric_real_batch.csv")
        csv_batch_real_file = open(val_metric_file, "w", newline="")
        csv_batch_real = csv.writer(csv_batch_real_file)
        csv_batch_real.writerow(["epoch","batch","accuracy","confusion_matrix_bin"," val_loss"])
        
        # val metric per batch real
        val_metric_file = os.path.join(log_save_path, "val_metric_fake_batch.csv")
        csv_batch_fake_file = open(val_metric_file, "w", newline="")
        csv_batch_fake = csv.writer(csv_batch_fake_file)
        csv_batch_fake.writerow(["epoch","batch","bin_accuracy","bin_recall", "bin_precision", "val_loss", "bin_IoU", "bin_dice", "bin_f1", "confusion_matrix_bin", "confusion_matrix_soft", "i_soft_dice", "i_soft_iou"])
        
        # files
        self.csv_file_batch = csv_file_batch
        self.csv_batch_fake_file = csv_batch_fake_file
        self.csv_batch_real_file = csv_batch_real_file
        self.csv_fake_epoch_file = csv_fake_epoch_file
        self.csv_real_epoch_file = csv_real_epoch_file
        self.csv_all_epoch_file = csv_all_epoch_file
        self.csv_file = csv_file
        # writer
        self.csv_writer_batch = csv_writer_batch
        self.csv_batch_fake = csv_batch_fake
        self.csv_writer = csv_writer
        self.csv_batch_real = csv_batch_real
        self.csv_fake_epoch = csv_fake_epoch
        self.csv_real_epoch = csv_real_epoch
        self.csv_all_epoch = csv_all_epoch

    def __del__(self):
        self.close_files()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_files()

    def return_writer(self):
        return self.csv_writer, self.csv_batch_fake, self.csv_batch_real, self.csv_real_epoch, self.csv_fake_epoch, self.csv_all_epoch, self.csv_writer_batch
    
    def close_files(self):
        self.csv_batch_fake_file.close()
        self.csv_batch_real_file.close()
        self.csv_fake_epoch_file.close()
        self.csv_real_epoch_file.close()
        self.csv_all_epoch_file.close()
        self.csv_file.close()
        self.csv_file_batch.close()