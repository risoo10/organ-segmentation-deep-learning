from utils.constants import *
import torch
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt


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

    def loadModalCheckpoint(self, model=None, loss=None, optimizer=None, epoch=None):
        path = f'{drive_dir}/torch/checkpoints/{self.DIR}{self.MODEL_NAME}.pth.tar'
        checkpoint = torch.load(path)

        if model != None:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer != None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer, checkpoint["loss"], checkpoint["epoch"]

    def loadTrainingData(self, columns):
        path = f'{drive_dir}/torch/logs/{self.MODEL_NAME}-training({",".join(columns)}).csv'
        return genfromtxt(path, delimiter=',')

    def plotTraining(self, loss, val_loss, lossName, title, epoch=None, legend=True):
        plt.plot(loss, 'b')
        plt.plot(val_loss, 'g')

        if epoch != None:
            plt.plot([epoch], val_loss[epoch], 'rx',  markersize=12)

        plt.xlabel('epoch')
        plt.ylabel(f'loss: {lossName}')
        if legend:
            plt.legend(['training', 'validation', 'min. validation loss'])
        plt.title(title)
