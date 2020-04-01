from utils.constants import *
import torch
import csv


class TrainingLogger():
    def __init__(self, MODEL_NAME="UNET-LIVER", DIR=""):
        self.MODEL_NAME = MODEL_NAME
        self.DIR = DIR

    def saveModelCheckpoint(self, model, loss, optimizer, epoch):
        path = f'{drive_dir}/torch/checkpoints/{self.DIR}{self.MODEL_NAME}.pth.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def saveTraingData(self, columns, data):
        path = f'{drive_dir}/torch/logs/{self.MODEL_NAME}-training({",".join(columns)}).csv'
        with open(path, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writerow(data)
