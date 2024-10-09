import numpy as np
import torch
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def visual(true, preds, actual = None, x_zigzag_data_true = None, y_zigzag_data_true = None, x_zigzag_data_pred = None, y_zigzag_data_pred = None, seg_len = None,  name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(20, 5))
    plt.plot(true, label='GroundTruth', linewidth=2)
    if actual is not None:
        plt.plot(actual, label='Actual', linewidth=3)
    
    if seg_len is not None and y_zigzag_data_true is not None:
        y_seg = [min(y_zigzag_data_true), max(y_zigzag_data_true)]
        plt.plot([seg_len, seg_len], y_seg, label='current', linewidth=2)
    
    if x_zigzag_data_true is not None and y_zigzag_data_true is not None:
        plt.plot(x_zigzag_data_true, y_zigzag_data_true, label='ZigZag True', linewidth=2)
    if x_zigzag_data_pred is not None and y_zigzag_data_pred is not None:
        plt.plot(x_zigzag_data_pred, y_zigzag_data_pred, label='ZigZag Pred', linewidth=2, linestyle='--')

    # Plot HL1, HL2
    number_HL_true = 0
    if x_zigzag_data_true is not None:
        for i, x in enumerate(x_zigzag_data_true):
            if x > seg_len:
                number_HL_true += 1
                plt.scatter(x_zigzag_data_true[i], y_zigzag_data_true[i], s=50, c='red', marker='o')
                price = str(round(y_zigzag_data_true[i], 2))
                plt.text(x_zigzag_data_true[i], y_zigzag_data_true[i], f"HL{number_HL_true}_{price}", fontsize=12)
            if number_HL_true == 2:
                break
    number_HL_pred = 0
    if x_zigzag_data_pred is not None:
        for i, x in enumerate(x_zigzag_data_pred):
            if x > seg_len:
                number_HL_pred += 1
                plt.scatter(x_zigzag_data_pred[i], y_zigzag_data_pred[i], s=50, c='red', marker='o')
                price = str(round(y_zigzag_data_pred[i], 2))
                plt.text(x_zigzag_data_pred[i], y_zigzag_data_pred[i], f"HL{number_HL_pred}_{price}", fontsize=12)
            if number_HL_pred == 2:
                break

    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean