
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import math

from utils import *
from models.ML_WKN_BiGRU_MSA import ML_WKN_BiGRU_MSA
from dataset_loader import CustomDataSet

def two_stage_train(Learning_Validation, hp, sX, work_condition):
    # random seed setup
    random.seed(42)
    np.random.seed(0)
    setup_seed(20)
    # access to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameter setup
    # sX = SSO_hp_trans(iX)
    batch_size = hp[0]
    num_epochs = hp[1]
    Learning_set = Learning_Validation[0]
    Validation_set = Learning_Validation[1]
    Validation_type = Learning_Validation[2]
    learning_rate = sX[0]/100000 # SSO update learning_rate, original = 0.001
    twist_point = sX[11]/100
    slope = sX[12]/100

