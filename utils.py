import pandas as pd
import numpy as np
import torch
import random
import os

def set_seed(seed=1):
    # seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_search_strings(feature_mode, selected_feature):
    if feature_mode == 'gaze-only':
        if selected_feature == 'all':
            search_strings = [' gaze']
        elif selected_feature == 'w/o_angle':
            search_strings = [' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z']
        elif selected_feature == 'all_with_lmk':
            search_strings = [' gaze', ' eye_lmk']
        elif selected_feature == 'w/o_angle_with_lmk':
            search_strings = [' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' eye_lmk']
    elif feature_mode == 'pose-only':
        if selected_feature == 'all':
            search_strings = [' pose']
        elif selected_feature == 'pose_T':
            search_strings = [' pose_T']
        elif selected_feature == 'pose_R':
            search_strings = [' pose_R']
    elif feature_mode == 'facial-only':
        if selected_feature == 'all':
            search_strings = [' x', ' y', ' X', ' Y', ' Z', ' eye_lmk']
        elif selected_feature == 'all_w/o_eye_lmk':
            search_strings = [' x', ' y', ' X', ' Y', ' Z']
    return search_strings

def read_data(root_path, label_path, labels, feature_mode, selected_feature):
    all_frame_feat = {}
    all_labels = {}
    # search_strings = [' x', ' y', ' X', ' Y', ' Z', ' eye_lmk']
    search_strings = get_search_strings(feature_mode, selected_feature)
    pattern = '(^' + "|^".join(search_strings)+')'
    for i in range(len(labels)):
        vid_id = labels.iloc[i,0]
        feat_path = os.path.join(root_path, vid_id+'_'+'video' + '.csv')
        feat_list = pd.read_csv(feat_path)
        feat_list = feat_list.filter(regex = pattern)
        feat_arr = np.array(feat_list).squeeze()
        feat_arr = feat_arr[-90:]
        
        label = labels.iloc[i,1]

        # frame_feat = feat_arr[-1]
        all_frame_feat[vid_id] = feat_arr
        all_labels[vid_id] = label  
    nun_bc_path = os.path.join(label_path, 'bc_detection_train.csv')
    nun_bc = pd.read_csv(nun_bc_path)
    for i in range(len(nun_bc)):
        if nun_bc.iloc[i,1] != 1:
            vid_id = nun_bc.iloc[i,0]
            feat_path = os.path.join(root_path, vid_id+'_'+'video' + '.csv')
            feat_list = pd.read_csv(feat_path)
            feat_list = feat_list.filter(regex = pattern)
            feat_arr = np.array(feat_list).squeeze()
            feat_arr = feat_arr[-90:]
            all_frame_feat[vid_id] = feat_arr
    return all_frame_feat, feat_arr.shape[1]

def get_det_pairs(df, nun_bc, num_pair):
    num_same_speaker = []
    ## search pairs
    name_pair = []
    spk2pairs = dict()
    for i in range(len(df)):
        if  df.label[i] == '-nan(ind)':
            df.label[i] = 0

        if df.speakers[i] not in spk2pairs:
            df_tmp = nun_bc[nun_bc.speakers==df.speakers[i]]
            num_same_speaker.append(len(df_tmp))
            if len(df_tmp) < num_pair:
                pairs = df_tmp.sample(len(df_tmp)).id.values
            else:
                pairs = df_tmp.sample(num_pair).id.values
            
            name_pair.append('-'.join(pairs))
            # print("3")
            spk2pairs[df.speakers[i]] = '-'.join(pairs)
        else:
            name_pair.append(spk2pairs[df.speakers[i]])


    df['name_pair'] = name_pair
    df = df.reset_index(drop=True)
    return df, num_same_speaker

def get_agreement_pairs(df, num_pair, neu=0, edge=0.1):
    num_same_speaker = []
    ## search pairs
    name_pair = []
    spk2pairs = dict()
    for i in range(len(df)):
        if  df.label[i] == '-nan(ind)':
            df.label[i] = 0

        if float(df.label[i]) <= neu+edge and float(df.label[i]) >= neu-edge :
            name_pair.append(df.id[i])
        else:
            if df.speakers[i] not in spk2pairs:
                df_tmp = df[df.speakers[i]==df.speakers]
                num_same_speaker.append(len(df_tmp))
                df_tmp = df_tmp[(df_tmp['label'] >= neu-edge) & (df_tmp['label'] <= neu+edge)]
                if len(df_tmp) < num_pair:
                    pairs = df_tmp.sample(len(df_tmp)).id.values
                else:
                    pairs = df_tmp.sample(num_pair).id.values
                
                name_pair.append('-'.join(pairs))
                spk2pairs[df.speakers[i]] = '-'.join(pairs)
            else:

                name_pair.append(spk2pairs[df.speakers[i]])


    df['name_pair'] = name_pair
    df = df.reset_index(drop=True)
    return df, num_same_speaker

def get_data(label_path, labels, mode, edge, num_pair, neu=0):      # edge_list = [0,0.1,0.2,0.3,0.4,0.5]; neu_list = [-1, 0, 0.25]  
    set_seed(seed=1)
    df = labels
    df['speakers'] = [s[-10:] for s in df.id]
    if neu == -1:
        # nun_bc = pd.read_csv('../../KZ/multimediate/backchannel_det/old_features/labels/bc_detection_'+'train.csv')
        nun_bc = pd.read_csv(os.path.join(label_path, 'bc_detection_'+'train.csv'))
        #nun_bc = pd.read_csv('../no-bc_speaker_labels.csv')
        nun_bc = nun_bc[nun_bc.label != 1]
        nun_bc = nun_bc.reset_index(drop=True)
        print(len(nun_bc))
        nun_bc['speakers'] = [s[-10:] for s in nun_bc.id]
        df, num_same_speaker = get_det_pairs(df,nun_bc, num_pair)
    else:
        df, num_same_speaker = get_agreement_pairs(df, num_pair, neu, edge)

    return df, num_same_speaker

def perturb_tensor_by_label_distance(
    x,
    labels,                 # shape (B,) or (B,1)
    mode=0,                 # 0~4
    noise_std=0.05,         # base noise
    drop_prob=0.1,          # base dropout
    noise_center=0.25,      # 想拟合的中心 label 值
    max_dyn_noise=0.1,
    min_drop=0.1,
    max_drop=0.5
):
    """
    Add perturbation based on how far labels are from a target center (default=0.25).

    The further the label from center, the stronger the noise and dropout.
    """
    if mode == 0:
        return x
    
    # with torch.no_grad():
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)

    # Compute |label - center|
    label_distance = torch.abs(labels - noise_center)  # shape (B, 1)

    # Normalize to [0, 1]
    norm_dist = label_distance / (label_distance.max() + 1e-6)

    # Expand to match x shape
    while norm_dist.dim() < x.dim():
        norm_dist = norm_dist.unsqueeze(-1)

    # --- Noise ---
    if mode in [1, 3]:  # 动态 noise
        dyn_std = torch.clamp(noise_std * norm_dist, max=max_dyn_noise)
        x = x + torch.randn_like(x) * dyn_std
        # print("dy std")
    elif mode in [2, 4]:  # 固定 noise
        x = x + torch.randn_like(x) * noise_std
        # print("fixed std")
    # mode 0 → no noise

    # --- Dropout ---
    if mode in [1, 4]:  # 固定 dropout
        mask = (torch.rand_like(x) > drop_prob).float()
        # print("fixed prob")
    elif mode in [2, 3]:  # 动态 dropout
        dyn_drop_prob = max_drop - norm_dist * (max_drop - min_drop)
        mask = (torch.rand_like(x) > dyn_drop_prob).float()
        # print("dy prob")

    x = x * mask

    return x