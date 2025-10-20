import numpy as np
import pandas as pd
from sklearn import metrics
import time
import random
import os
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.backends import cudnn
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from torch.utils.data import Dataset
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error
import csv
import torch.nn.functional as F
from network import Transformer_Encoder, GRU_Classifier, Translator, Frame_Level_Encoder, FrameLevelTranslator
from utils import read_data, get_data
from dataset import my_dataset

# Ensure we don't use interactive mode when running as script
plt.switch_backend('agg')
np.set_printoptions(suppress=True)

def set_seed(seed=1):
    # seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)


def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate concordance correlation coefficient (CCC) and other metrics"""
    # print(y_true)
    # print(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0, 1]
    
    # Calculate Pearson correlation
    pearson_corr, p_value_pearson = pearsonr(y_true, y_pred)
    
    # Calculate Kendall correlation
    kendall_corr, p_value_kendall = kendalltau(y_true, y_pred)

    # Calculate CCC
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    
    return {
        "CCC": ccc,
        "Pearson Correlation": pearson_corr,
        "Pearson p-value": p_value_pearson,
        "Kendall Correlation": kendall_corr,
        "Kendall p-value": p_value_kendall
    }

def get_input_dim(feature_mode, selected_feature):
    if feature_mode == 'gaze-only':
        if selected_feature == 'all':
            input_dim = 8
        elif selected_feature == 'w/o_angle':
            input_dim = 6
        elif selected_feature == 'all_with_lmk':
            input_dim = 288
        elif selected_feature == 'w/o_angle_with_lmk':
            input_dim = 286
    elif feature_mode == 'pose-only':
        if selected_feature == 'all':
            input_dim = 6
        elif selected_feature == 'pose_T':
            input_dim = 3
        elif selected_feature == 'pose_R':
            input_dim = 3
    elif feature_mode == 'facial-only':
        if selected_feature == 'all':
            input_dim = 620
        elif selected_feature == 'all_w/o_eye_lmk':
            input_dim = 340
    return input_dim

def train_and_evaluate(args):
    """Main training and evaluation function"""
    # Set seed for reproducibility
    # set_seed(args.seed) 

    # # Load data
    print(args.is_visualbc)
    if args.is_visualbc:
        print("non-speaker~~~~~")
        print("is_visualbc =", args.is_visualbc, type(args.is_visualbc))
        train_labels = pd.read_csv("../ASD_agreement_speaker_threshold=7_conf=0.2_labels_train.csv")
        train_labels = train_labels[train_labels['label'] == 0].reset_index(drop=True)
        df = pd.read_csv(os.path.join(args.label_path, 'bc_agreement_train.csv'))
        train_labels['label'] = train_labels['id'].map(df.set_index('id')['label'])
    else:
        train_labels = pd.read_csv(os.path.join(args.label_path, 'bc_agreement_train.csv'))
    if args.is_visualbc:
        val_labels = pd.read_csv("../ASD_agreement_speaker_threshold=7_conf=0.2_labels_val.csv")
        val_labels = val_labels[val_labels['label'] == 0].reset_index(drop=True)
        df = pd.read_csv(os.path.join(args.label_path, 'bc_agreement_val.csv'))
        val_labels['label'] = val_labels['id'].map(df.set_index('id')['label'])
    else:
        val_labels = pd.read_csv(os.path.join(args.label_path, 'bc_agreement_val.csv'))
    print(f"Total train samples: {len(train_labels)}")
    print(f"Total val samples: {len(val_labels)}")

    input_dim = get_input_dim(args.feature_mode, args.selected_feature)
    # Create output directories
    for path in [args.model_dir, args.result_dir, args.pred_dir, args.pic_dir]:
        os.makedirs(path, exist_ok=True)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create models based on mode
    # Base model with Transformer Encoder
    E = Transformer_Encoder(
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        noise = args.sle_noise,
        is_dynoise = args.is_dynoise,
        noise_std = args.noise_std,
        drop_prob = args.drop_prob
    ).to(device)
    C = GRU_Classifier(input_dim=args.hidden_dim, num_classes=1).to(device)

    if args.mode == 'isnet':
        # ISNet model with additional translator
        E2 = Transformer_Encoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            noise=False,
            is_dynoise = args.is_dynoise,
            noise_std = args.noise_std,
            drop_prob = args.drop_prob
        ).to(device)
        # C2 = GRU_Classifier(input_dim=args.hidden_dim*2, hidden_dim=args.hidden_dim, num_classes=1).to(device)
        C2 = GRU_Classifier(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_classes=1,ave_method=args.ave_method).to(device)
        T = Translator(args.hidden_dim, args.hidden_dim//2, args.hidden_dim, args.translator_mode).to(device)
        # T = origin_Translator(args.hidden_dim).to(device)

        E_frame_level = Frame_Level_Encoder(input_dim=input_dim, hidden_dim=args.hidden_dim, noise = args.fle_noise,is_dynoise = args.is_dynoise,
                                            noise_std = args.noise_std,drop_prob = args.drop_prob).to(device)
        C_frame_level = GRU_Classifier(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_classes=1).to(device)

        E2_frame_level = Frame_Level_Encoder(input_dim=input_dim, hidden_dim=args.hidden_dim, noise = False,is_dynoise = args.is_dynoise,
                                            noise_std = args.noise_std,drop_prob = args.drop_prob).to(device)

        # T_frame_level = origin_Translator(args.hidden_dim).to(device)
        T_frame_level = FrameLevelTranslator(args.hidden_dim, args.hidden_dim//2,num_layers=3).to(device)
    # Initialize optimizers
    optimizer_E = torch.optim.SGD(E.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
    optimizer_C = torch.optim.SGD(C.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
    
    # Initialize schedulers
    scheduler_E = lr_scheduler.StepLR(optimizer_E, step_size=args.step_size, gamma=args.gamma)
    scheduler_C = lr_scheduler.StepLR(optimizer_C, step_size=args.step_size, gamma=args.gamma)    

    if args.mode == 'isnet':
        # Initialize optimizers
        optimizer_E2 = torch.optim.SGD(E2.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        scheduler_E2 = lr_scheduler.StepLR(optimizer_E2, step_size=args.step_size, gamma=args.gamma)
        optimizer_C2 = torch.optim.SGD(C2.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        optimizer_T = torch.optim.SGD(T.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        
        # Initialize schedulers
        scheduler_C2 = lr_scheduler.StepLR(optimizer_C2, step_size=args.step_size, gamma=args.gamma)
        scheduler_T = lr_scheduler.StepLR(optimizer_T, step_size=args.step_size, gamma=args.gamma)

        optimizer_E_frame_level = torch.optim.SGD(E_frame_level.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        optimizer_C_frame_level = torch.optim.SGD(C_frame_level.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        optimizer_T_frame_level = torch.optim.SGD(T_frame_level.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        optimizer_E2_frame_level = torch.optim.SGD(E2_frame_level.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)

        scheduler_E_frame_level = lr_scheduler.StepLR(optimizer_E_frame_level, step_size=args.step_size, gamma=args.gamma)
        scheduler_C_frame_level = lr_scheduler.StepLR(optimizer_C_frame_level, step_size=args.step_size, gamma=args.gamma)
        scheduler_T_frame_level = lr_scheduler.StepLR(optimizer_T_frame_level, step_size=args.step_size, gamma=args.gamma)
        scheduler_E2_frame_level = lr_scheduler.StepLR(optimizer_E2_frame_level, step_size=args.step_size, gamma=args.gamma)

    # Initialize loss function
    if args.loss_function == 'MSE':
        loss_class = nn.MSELoss().to(device)
    elif args.loss_function == 'MAE':
        loss_class = nn.L1Loss().to(device)
    loss_dist = nn.L1Loss().to(device)

    loss_train, loss_test = [], []
    loss2_train, loss2_test = [], []
    metric_train, metric_test = [], []
    accuracy_train, accuracy_test = [], []
    mse_train, mse_test = [], []
    train_all_feat, val_all_feat, df_train, df_val, datasets,loaders = None, None,None,None,None,None

    # Process features
    train_all_feat, _ = read_data(args.feat_dir, args.label_path, train_labels, args.feature_mode, args.selected_feature)
    val_all_feat, _ = read_data(args.feat_dir, args.label_path, val_labels, args.feature_mode, args.selected_feature)
    
    print(f"Feature dimension: {input_dim}")
    
    # Prepare datasets
    df_train, train_speakers = get_data(args.label_path, train_labels, 'train', args.edge, args.num_pair, args.neu)
    df_val, val_speakers = get_data(args.label_path, val_labels, 'val', args.edge, args.num_pair, args.neu)
    
    print(f"Processed train samples: {len(df_train)}")
    print(f"Processed val samples: {len(df_val)}")
    
    # Print statistics
    print("Train speakers distribution:", Counter(train_speakers))
    print("Val speakers distribution:", Counter(val_speakers))
    
    # Create datasets and loaders
    datasets = {
        'train': my_dataset(df_train, train_all_feat, args.num_pair),
        'test': my_dataset(df_val, val_all_feat, args.num_pair)
    }
    
    loaders = {
        'train': Data.DataLoader(
            dataset=datasets['train'], 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,
        ),
        'test': Data.DataLoader(
            dataset=datasets['test'], 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,
        )
    }
    
    # Train model
    best_val_mse = float('inf')
    best_metrics = None
    early_stop_counter = 0
    
    # Save model IDs for filename generation
    model_id = f"{args.mode}_{args.feature_mode}_{args.selected_feature}_neu{args.neu}_edge{args.edge}_lr{args.lr}_bs{args.batch_size}_nl{args.num_layers}_nh{args.num_heads}"
    
    model_path = os.path.join(args.model_dir, f"{model_id}.pt")
    result_path = os.path.join(args.result_dir, f"{model_id}.csv")
    plot_path = os.path.join(args.pic_dir, f"{model_id}.png")
    
    for epoch in range(args.epochs):
        # Training phase
        E.train()
        C.train()
        E2.train()
        C2.train()
        T.train()
        E_frame_level.train()
        C_frame_level.train()
        T_frame_level.train()
        E2_frame_level.train()
        
        train_start = time.time()
        train_loss = 0
        train_mse = 0
        train_samples = 0
        loss_tr, loss2_tr = 0.0, 0.0
        pred_all, actu_all = [], []
        
        for step, (xs, xs_pairs, ys) in enumerate(loaders['train']):
            xs, ys = xs.to(device), ys.to(device)
            
            ## train E_frame_level
            features = E_frame_level(xs, ys)
            out, _ = C_frame_level(features)
            loss_frame = loss_class(out.squeeze(1), ys)
            E_frame_level.zero_grad()
            C_frame_level.zero_grad()
            loss_frame.backward()
            optimizer_E_frame_level.step()
            optimizer_C_frame_level.step()
            
            ## train E2_frame_level
            out_E_pairs = []
            for xs_pair in xs_pairs:
                # out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs.append(E2_frame_level(xs_pair.to(device).squeeze(1)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            loss3_frame = 0
            for i in range(args.num_pair-1):
                for j in range(i+1, args.num_pair):
                    loss3_frame += loss_dist(out_E_pairs[:,i], out_E_pairs[:,j])
            E2_frame_level.zero_grad()
            loss3_frame.backward()
            optimizer_E2_frame_level.step()

            ## train T_frame
            out_E_pairs = []
            for xs_pair in xs_pairs:
                # out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs.append(E2_frame_level(xs_pair.to(device).squeeze(1)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
            out_T = T_frame_level(E_frame_level(xs,ys))
            loss2_frame = loss_dist(out_T, out_E_pair)
            T_frame_level.zero_grad()
            loss2_frame.backward()
            optimizer_T_frame_level.step()

            xs = E_frame_level(xs, ys)-T_frame_level(E_frame_level(xs, ys))
            xs = xs.detach()
            xs = xs.to(device)

            ## train E
            features = E(xs, ys)
            out, _ = C(features)
            
            # Calculate loss
            loss_E = loss_class(out.squeeze(1), ys)
            E.zero_grad()
            C.zero_grad()
            loss_E.backward()
            
            # Update weights
            optimizer_E.step()
            optimizer_C.step()

            ## Train E2
            out_E_pairs = []
            for xs_pair in xs_pairs:
                # out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs.append(E2(xs_pair.to(device).squeeze(1)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            loss3 = 0
            for i in range(args.num_pair-1):
                for j in range(i+1, args.num_pair):
                    loss3 += loss_dist(out_E_pairs[:,i], out_E_pairs[:,j])
            E2.zero_grad()
            loss3.backward()
            optimizer_E2.step()

            ## Train T
            out_E_pairs = []
            for xs_pair in xs_pairs:
                # out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs.append(E2(xs_pair.to(device).squeeze(1)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
            out_T = T(E(xs,ys))
            loss2 = loss_dist(out_T, out_E_pair)
            T.zero_grad()
            loss2.backward()
            optimizer_T.step()

            ## Train C2
            out_E = E(xs,ys)
            out_E = out_E - T(out_E)
            out, features = C2(out_E)
            loss = loss_class(out.squeeze(1), ys)
            C2.zero_grad()
            loss.backward()
            optimizer_C2.step()
            
            # Update statistics
            pred = out.cpu().data
            actu = ys.cpu().data.numpy()
            pred_all = pred_all + list(pred)
            actu_all = actu_all + list(actu)

            loss_tr += loss.cpu().item()
            loss2_tr += loss2.cpu().item()
        loss_tr = loss_tr / len(loaders['train'].dataset)
        loss2_tr = loss2_tr / len(loaders['train'].dataset)
        
        pred_all, actu_all = np.array(pred_all), np.array(actu_all)
        mse_tr = mean_squared_error(actu_all, pred_all)
        loss_train.append(loss_tr)
        loss2_train.append(loss2_tr)
        mse_train.append(mse_tr)
        print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| mse: %.3f' % mse_tr, '| lr: %.5f' % optimizer_E.param_groups[0]['lr']) 
        
        with torch.no_grad():
            # Validation phase
            E.eval()
            C.eval()
            E2.eval()
            C2.eval()
            T.eval()
            T_frame_level.eval()
            E2_frame_level.eval()
            E_frame_level.eval()
            C_frame_level.eval()

            loss_te, loss2_te = 0.0, 0.0
            pred_all, actu_all = [], []
        
            for step, (xs, xs_pairs, ys) in enumerate(loaders['test']):
                xs, ys = xs.to(device), ys.to(device)

                xs = E_frame_level(xs, ys)-T_frame_level(E_frame_level(xs, ys))
                out_E_pairs = []
                for xs_pair in xs_pairs:
                    # out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                    out_E_pairs.append(E2(xs_pair.to(device).squeeze(1)).unsqueeze(1))
                out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
                out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
                out_T = T(E(xs))
                loss2 = loss_dist(out_T, out_E_pair)

                out_E = E(xs)
                out_E = out_E - T(out_E)
                out, features = C2(out_E)
                loss = loss_class(out.squeeze(1), ys)


                pred = out.cpu().data
                # print(pred)
                actu = ys.cpu().data.numpy()
                pred_all = pred_all + list(pred)
                actu_all = actu_all + list(actu)

                loss_te += loss.cpu().item()
                loss2_te += loss2.cpu().item()

        loss_te = loss_te / len(loaders['test'].dataset)
        loss2_te = loss2_te / len(loaders['test'].dataset)

        pred_all, actu_all = np.array(pred_all), np.array(actu_all)
        mse_te = mean_squared_error(actu_all, pred_all)
        metrics = concordance_correlation_coefficient(actu_all, pred_all.flatten())
        
        loss_test.append(loss_te)
        loss2_test.append(loss2_te)
        mse_test.append(mse_te)
        print('TEST :: Epoch: ', epoch, '| Loss: %.3f' % loss_te, '| mse: %.3f' % mse_te, '| lr: %.5f' % optimizer_C.param_groups[0]['lr'])

        scheduler_E.step()
        scheduler_C.step()
        scheduler_E2.step()
        scheduler_C2.step()
        scheduler_T.step()
        scheduler_T_frame_level.step()
        scheduler_E_frame_level.step()
        scheduler_E2_frame_level.step()
        scheduler_C_frame_level.step()
        
        # Check for improvement
        if mse_te < best_val_mse:
            best_val_mse = mse_te
            best_ccc = metrics['CCC']
            best_metrics = metrics
            if args.if_save:
                torch.save(E.state_dict(), os.path.join(args.model_dir, 'E.pth'))
                torch.save(C.state_dict(), os.path.join(args.model_dir, 'C.pth'))
                # if args.mode == 'isnet':
                torch.save(E2.state_dict(), os.path.join(args.model_dir, 'E2.pth'))
                torch.save(C2.state_dict(), os.path.join(args.model_dir, 'C2.pth'))
                torch.save(T.state_dict(), os.path.join(args.model_dir, 'T.pth'))
                torch.save(T_frame_level.state_dict(), os.path.join(args.model_dir, 'T_frame_level.pth'))
                torch.save(E_frame_level.state_dict(), os.path.join(args.model_dir, 'E_frame_level.pth'))
                torch.save(E2_frame_level.state_dict(), os.path.join(args.model_dir, 'E2_frame_level.pth'))
                torch.save(C_frame_level.state_dict(), os.path.join(args.model_dir, 'C_frame_level.pth'))
            # early_stop_counter = 0
            
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= args.early_stop:
        #         print(f"Early stopping triggered at epoch {epoch}")
        #         break
        
        # Print best results so far
        print(f"best_val_mse: {best_val_mse} best_ccc: {best_ccc} " +
              f"best Pearson Correlation: {best_metrics['Pearson Correlation']} " +
              f"best Pearson p-value: {best_metrics['Pearson p-value']} " +
              f"best Kendall Correlation: {best_metrics['Kendall Correlation']} " + 
              f"best Kendall p-value: {best_metrics['Kendall p-value']}")
    
    # Print final best results
    df = pd.Series({'mode':args.mode, 'translator':args.translator_mode,'model':'transformer','feature_mode':args.feature_mode, 'selected_feature':args.selected_feature,
                            'preprocessing': 'subtract','neu':args.neu, 'edge': args.edge,'hidden_dim':args.hidden_dim, 'batch_size':args.batch_size, 'lr':args.lr, 'num_layer':args.num_layers, 
                            'num_head':args.num_heads,'num_pair':args.num_pair,
                            'best_val_mse':best_val_mse, 'best_ccc': best_metrics['CCC'], 'best Pearson Correlation':best_metrics['Pearson Correlation'],
                            'best Pearson p-value':best_metrics['Pearson p-value'], 'best Kendall Correlation':best_metrics['Kendall Correlation'],
                            'best Kendall p-value':best_metrics['Kendall p-value']}).to_frame().T
    # dataframe.to_csv('csv/agreement.csv', index=False, sep=',')
    print(best_val_mse)
    
    
    # Return best results for logging
    return {
        'final_best_val_mse': best_val_mse,
        'final_best_ccc': best_metrics['CCC'],
        'final_best_Pearson_Correlation':best_metrics['Pearson Correlation'],
        'final_best_Pearson_p-value': best_metrics['Pearson p-value'],
        'final_best_Kendall_Correlation':best_metrics['Kendall Correlation'],
        'final_best_Kendall_p-value':best_metrics['Kendall p-value'],
        'model_id': model_id
    }

if __name__ == '__main__':
    # Parse command-line arguments

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("yes", "true", "t", "1" "True")


    parser = argparse.ArgumentParser(description='Single Modality Transformer for Agreement Prediction')
    
    # Data paths
    parser.add_argument('--label_path', type=str, 
                        default='/PATH/TO/YOUR/LABEL',
                        help='Path to labels')
    parser.add_argument('--feat_dir', type=str,
                        default='/PATH/TO/YOUR/FEATURE',
                        help='Directory containing feature files')
    
    # Output paths
    parser.add_argument('--model_dir', type=str, default='model/isnet_f&s_noise0',
                        help='Directory to save models')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory to save results')
    parser.add_argument('--pred_dir', type=str, default='pred',
                        help='Directory to save predictions')
    parser.add_argument('--pic_dir', type=str, default='picture',
                        help='Directory to save plots')
    parser.add_argument('--if_save', type=bool, default=False,
                        help='Directory to save logs')
    
    # Model parameters
    parser.add_argument('--mode', type=str, choices=['base', 'isnet'], default='isnet',
                        help='Model mode: base or isnet')
    parser.add_argument('--translator_mode', type=str, 
                        choices=['ED-LSTM', 'ED-GRU', 'ED-Transformer'], 
                        default='ED-LSTM',
                        help='Translator mode for ISNet')
    parser.add_argument('--feature_mode', type=str, 
                        choices=['facial-only', 'gaze-only', 'pose-only'], 
                        default='facial-only',
                        help='Feature modality to use')
    parser.add_argument('--selected_feature', type=str, default='all',
                        help='Selected features within the modality')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizers')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--early_stop', type=int, default=21,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--loss_function', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--is_dynoise', type=int,default=4)
    parser.add_argument('--is_visualbc', type=str2bool,default=True)
    parser.add_argument('--fle_noise',type=str2bool,default=True)
    parser.add_argument('--sle_noise',type=str2bool,default=True)
    parser.add_argument('--noise_std',type=float,default=0.05)
    parser.add_argument('--drop_prob',type=float,default=0.1)
    parser.add_argument('--ave_method',type=str,default='temp_attn_mean')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    
    # Data parameters
    parser.add_argument('--num_pair', type=int, default=4,
                        help='Number of data pairs to use')
    parser.add_argument('--neu', type=float, default=-1,
                        help='Neutral value for agreement pairs')
    parser.add_argument('--edge', type=float, default=0,
                        help='Edge threshold for agreement pairs')
    
    args = parser.parse_args()
    
    # Set global variables based on args
    # print(args.is_visualbc)
    # base_dir = args.base_dir
    # feat_dir = args.feat_dir
    
    # Run the experiment
    results = train_and_evaluate(args)
    
    # Print final results
    print(f"model_id: {results['model_id']}")
    print(f"final_best_val_mse: {results['final_best_val_mse']:.6f}")
    print(f"final_best_ccc: {results['final_best_ccc']:.6f}")
    print(f"final_best_Pearson_Correlation: {results['final_best_Pearson_Correlation']:.6f}")
    print(f"final_best_Pearson_p-value: {results['final_best_Pearson_p-value']:.6f}")
    print(f"final_best_Kendall_Correlation: {results['final_best_Kendall_Correlation']:.6f}")
    print(f"final_best_Kendall_p-value: {results['final_best_Kendall_p-value']:.6f}")