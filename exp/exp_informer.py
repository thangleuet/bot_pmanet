import joblib
import pandas as pd
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from exp.pattern_model import PatternModel
from models.model import Informer, InformerStack
from utils.timefeatures import time_features

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import tqdm

import warnings

warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     criterion =  nn.MSELoss()
    #     return criterion

    def _select_criterion(self):
        l1_loss = nn.SmoothL1Loss()
        # 创建MSE损失函数
        mse_loss = nn.L1Loss()

        # 定义一个函数，用于计算两个损失函数的平均值
        def criterion(input, target):
            # 计算L1损失
            loss1 = l1_loss(input, target)
            # 计算MSE损失
            loss2 = mse_loss(input, target)
            # 计算两个损失的平均值
            loss = torch.mean(torch.stack([loss1, loss2]))
            return loss

        return criterion

    def vali(self, train_data, vali_loader, criterion, epoch):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

            # innvert scaling
            if train_data.scaler:
                pred = train_data.inverse_transform(pred)
                true = train_data.inverse_transform(true)

            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()

            folder_result = f'test_results/{epoch}'

            if not os.path.exists(folder_result):
                os.makedirs(folder_result)

            if i % 1 == 0:
                if train_data.scaler:
                    batch_x = train_data.inverse_transform(batch_x)
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                true_max_idx = np.argmax(true[0, :, -1]) + input[0, :, -1].shape[0]
                true_min_idx = np.argmin(true[0, :, -1]) + input[0, :, -1].shape[0]

                pd_max_idx = np.argmax(pred[0, :, -1]) + input[0, :, -1].shape[0]
                pd_min_idx = np.argmin(pred[0, :, -1]) + input[0, :, -1].shape[0]

                mae = np.mean(np.abs(pred[0, :, -1] - true[0, :, -1]))

                name_image = f"{i}_{round(true_max_idx - pd_max_idx, 2)}_{round(true_min_idx - pd_min_idx, 2)}_{round(mae, 2)}"
                visual(gt, pd, name = os.path.join(folder_result, name_image + '.png'))


        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = []  # 存储训练的loss值
        vali_losses = []  # 存储验证的loss值
        test_losses = []  # 存储测试的loss值

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []  # 存储每个epoch的训练的loss值

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())  # 将每个batch的loss值添加到train_loss列表中

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 计算每个epoch的训练的loss值的平均值
            train_losses.append(train_loss)  # 将每个epoch的训练的loss值添加到train_losses列表中
            vali_loss = self.vali(train_data, vali_loader, criterion, epoch)  # 计算每个epoch的验证的loss值
            vali_losses.append(vali_loss)  # 将每个epoch的验证的loss值添加到vali_losses列表中
    
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch + 1}.pth')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def analyze_model(self):
        self.model.load_state_dict(torch.load(os.path.join('weight', 'checkpoint_97.pth'), map_location=self.device))
        self.model.eval()

        df_raw_test = pd.read_csv(r"test/exness_xau_usd_m30_2024_test.csv")
        # df_raw_test.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'volume'] 
        df_stamp = df_raw_test[['Date']]
        # data_stamp = time_features(pd.to_datetime(df_stamp.Date), freq='h')

        df_stamp['date'] = pd.to_datetime(df_stamp.Date)
        data_stamp = time_features(df_stamp, timeenc=0, freq=self.args.freq)

        self.data = df_raw_test[['Open', 'High', 'Low', 'volume', 'Close']].values
        train_scaler = joblib.load('scaler.pkl')

        train_data, train_loader = self._get_data(flag='train')
        train_scaler = train_data.scaler

        if train_scaler is not None:
            self.data = train_scaler.transform(self.data)

        list_mae = []
        list_pattern_predict = []

        for i, data_candle in tqdm.tqdm(enumerate(self.data)):
            mae, list_zigzag_true, list_zigzag_pred, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred = self.inference(i, data_stamp, train_scaler)
        #     list_mae.append(mae)
        #     if len(list_zigzag_pred) > 0:
        #         if len(list_pattern_predict) == 0:
        #                 pattern_model = PatternModel(list_zigzag_pred, list_zigzag_true, i, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred)
        #                 list_pattern_predict.append(pattern_model)
        #         else:
        #             last_pattern_model = list_pattern_predict[-1]
        #             if last_pattern_model.list_zigzag_pred[0][2] == list_zigzag_pred[0][2]:
        #                 last_pattern_model.confirm_count += 1
        #                 last_pattern_model.list_zigzag_true = list_zigzag_true
        #                 last_pattern_model.list_zigzag_pred = list_zigzag_pred
        #                 last_pattern_model.index_candle = i
        #                 last_pattern_model.path_image = path_image
        #                 last_pattern_model.actual = actual
        #                 last_pattern_model.pred = pred
        #                 last_pattern_model.gt = gt
        #                 last_pattern_model.x_zigzag_data_true = x_zigzag_data_true
        #                 last_pattern_model.y_zigzag_data_true = y_zigzag_data_true
        #                 last_pattern_model.x_zigzag_data_pred = x_zigzag_data_pred
        #                 last_pattern_model.y_zigzag_data_pred = y_zigzag_data_pred

        #             else:
        #                 pattern_model = PatternModel(list_zigzag_pred, list_zigzag_true, i, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred)
        #                 list_pattern_predict.append(pattern_model)

        # count_confirm_pattern = 0     
        # count_confirm_corection = 0              
        # for pattern_model in list_pattern_predict:
        #     if pattern_model.confirm_count >= 5 and len(pattern_model.list_zigzag_pred) > 1:
        #         count_confirm_pattern += 1
        #         # if len(pattern_model.list_zigzag_pred) > 1 and len(pattern_model.list_zigzag_true) > 1:
        #         if pattern_model.list_zigzag_pred[0][2] == pattern_model.list_zigzag_true[0][2]:
        #             count_confirm_corection += 1
        #             path_image = pattern_model.path_image
        #             visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)
        #         else:
        #             path_image = pattern_model.path_image.replace('correction', 'fail')
        #             if not os.path.exists(os.path.dirname(path_image)): 
        #                 os.makedirs(os.path.dirname(path_image)) 
        #             visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)
        #         # else:
        #         #     path_image = pattern_model.path_image.replace('correction', 'fail')
        #         #     if not os.path.exists(os.path.dirname(path_image)): 
        #         #         os.makedirs(os.path.dirname(path_image))
        #         #     visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)

        #     print(f'Index: {pattern_model.index_candle} count: {pattern_model.confirm_count} list_type_pred: {pattern_model.list_zigzag_pred} list_type_true: {pattern_model.list_zigzag_true}')
        
        # print(f'Count confirm pattern: {count_confirm_pattern}/{len(list_pattern_predict)}')
        # print(f'Count confirm corection: {count_confirm_corection}/{count_confirm_pattern}')
        

    def inference(self, index_candle, data_stamp, train_scaler):
        s_end = index_candle
        s_begin = s_end - self.args.seq_len
        r_begin = s_end - self.args.label_len
        r_end = index_candle + self.args.pred_len
        if index_candle <= self.args.seq_len or r_end >= len(self.data):
            return None, [], [], None, None, None, None, None, None, None, None

        batch_x = self.data[s_begin:s_end]
        batch_y = self.data[r_begin:r_end]
        batch_x_mark = data_stamp[s_begin:s_end]
        batch_y_mark = data_stamp[r_begin:r_end]

        actual = train_scaler.inverse_transform(self.data[s_begin:r_end])[:, -1]


        batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(self.device)
        batch_y = torch.from_numpy(batch_y).float().unsqueeze(0).to(self.device)
        batch_x_mark = torch.from_numpy(batch_x_mark).float().unsqueeze(0).to(self.device)
        batch_y_mark = torch.from_numpy(batch_y_mark).float().unsqueeze(0).to(self.device)

        pred, true = self._process_one_batch(
                train_scaler, batch_x, batch_y, batch_x_mark, batch_y_mark)
        
        pred = train_scaler.inverse_transform(pred)
        true = train_scaler.inverse_transform(true)

        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()

        folder_path = 'results'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if visual:
            batch_x = train_scaler.inverse_transform(batch_x)
            input = batch_x.detach().cpu().numpy()
            gt = np.concatenate((input[0, :, -1], true[0, :self.args.pred_len, -1]), axis=0)
            pd = np.concatenate((input[0, :, -1], pred[0, :self.args.pred_len, -1]), axis=0)
            true_max_idx = np.argmax(true[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]
            true_min_idx = np.argmin(true[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]

            pd_max_idx = np.argmax(pred[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]
            pd_min_idx = np.argmin(pred[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]

            mae = np.mean(np.abs(pred[0, :self.args.pred_len, -1] - true[0, :self.args.pred_len, -1]))

            list_y_candle_true = np.concatenate((input[0, :, ], true[0, :, ]), axis=0)
            list_y_candle_pred = np.concatenate((input[0, :, ], pred[0, :, ]), axis=0)

            x_zigzag_data_true, y_zigzag_data_true, type_zigzag_data_true = self.analys_zigzag_data(list_y_candle_true)
            x_zigzag_data_pred, y_zigzag_data_pred, type_zigzag_data_pred = self.analys_zigzag_data(list_y_candle_pred)
            
            list_zigzag_true = []
            for i, x in enumerate(x_zigzag_data_true):
                if x > self.args.seq_len:
                    list_zigzag_true.append([x, y_zigzag_data_true[i], type_zigzag_data_true[i]])

            list_zigzag_pred = []
            for i, x in enumerate(x_zigzag_data_pred):
                if x > self.args.seq_len:
                    list_zigzag_pred.append([x, y_zigzag_data_pred[i], type_zigzag_data_pred[i]])

            if len(list_zigzag_true) > 0 and len(list_zigzag_pred) > 0:
                if list_zigzag_true[0] == list_zigzag_pred[0]:
                    correction_trend = 1
                else:
                    correction_trend = 0
            else:
                correction_trend = 0

            name_image = f"{index_candle}_{round(true_max_idx - pd_max_idx, 2)}_{round(true_min_idx - pd_min_idx, 2)}_{correction_trend}_{round(mae, 2)}"
            visual(gt, pd, actual, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred, self.args.seq_len, os.path.join(folder_path, name_image + '.png'))
            
            path_correction_image = os.path.join(folder_path, 'correction')
            if not os.path.exists(path_correction_image):
                os.makedirs(path_correction_image)
            path_image = os.path.join(path_correction_image, name_image + '.png')

        return mae, list_zigzag_true, list_zigzag_pred, gt, pd, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred
    def percent(self, start, stop):
        if start != 0:
            percent = float(((float(start) - float(stop)) / float(start))) * 100
            if percent > 0:
                return percent
            else:
                return abs(percent)
        return 1
    def analys_zigzag_data(self, list_y_candle):
        percent_filter = 0.4
        candle_timedata_pass = 3
        last_zigzag = 50
        list_zigzag = []
        for candle_timedata in range(len(list_y_candle)):
            candle_data = list_y_candle[candle_timedata]
            if len(list_zigzag) == 0:
                list_zigzag = [[candle_timedata, candle_data[2],'low'], [candle_timedata, candle_data[1],'high']]
            
            if self.percent(list_zigzag[0][1], list_zigzag[1][1]) < percent_filter:
                if list_zigzag[0][2] == "low":
                    if list_zigzag[0][1] > candle_data[2]:
                        list_zigzag.pop(0)
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])
                    elif list_zigzag[1][1] < candle_data[1]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                elif list_zigzag[0][2] == "high":
                    if list_zigzag[0][1] < candle_data[1]:
                        list_zigzag.pop(0)
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                    elif list_zigzag[1][1] > candle_data[2]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])

            else:
                if list_zigzag[-1][2] == "low":
                    if list_zigzag[-1][1] > candle_data[2]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])
                    elif (
                        self.percent(list_zigzag[-1][1], candle_data[1])
                        > percent_filter
                    ):
                        if candle_timedata - list_zigzag[-1][0] >= candle_timedata_pass:
                            list_zigzag.append(
                                [candle_timedata, candle_data[1], "high"]
                            )

                elif list_zigzag[-1][2] == "high":
                    if list_zigzag[-1][1] < candle_data[1]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                    elif (
                        self.percent(list_zigzag[-1][1], candle_data[2])
                        > percent_filter
                    ):
                        if candle_timedata - list_zigzag[-1][0] >= candle_timedata_pass:
                            list_zigzag.append(
                                [candle_timedata, candle_data[2], "low"]
                            )
                    # elif self.percent(list_zigzag[-1][1], candle_data[2]) < percent_filter and candle_timedata - list_zigzag[-1][0] >= process_config.config[f"{self.time_frame}"]["zigzag"]["candle_timedata_update_zigzag"]:
                    #     list_zigzag.append(
                    #             [candle_timedata, candle_data[2], "low"]
                    #         )

        if len(list_zigzag) == 2:
            if (
                self.percent(list_zigzag[0][1], list_zigzag[1][1])
                >= percent_filter
            ):
                x_data = [x[0] for x in list_zigzag]
                y_data = [x[1] for x in list_zigzag]
                type_data = [x[2] for x in list_zigzag]
            else:
                x_data, y_data, type_data = [], [], []
        else:
            x_data = [x[0] for x in list_zigzag]
            y_data = [x[1] for x in list_zigzag]
            type_data = [x[2] for x in list_zigzag]
                
        list_zigzag = list_zigzag[-last_zigzag:]
        return x_data, y_data, type_data

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse = metric(preds, trues)
        print('mse:{}, mae:{}, rmse: {}, mape: {}, rse: {}'.format(mse, mae, rmse, mape, rse))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,0:].to(self.device)

        return outputs, batch_y
