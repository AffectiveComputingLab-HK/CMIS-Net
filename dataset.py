import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch

class my_dataset(Dataset):
    def __init__(self, names_labels, all_feat, num_pair):
        self.names_labels = names_labels
        self.num_pair = num_pair

        self.pair_names = np.array(names_labels['name_pair'])
        self.names = np.array(names_labels['id'])
        self.labels = np.array(names_labels['label'])
        self.all_feat = all_feat
    
    def z_score_normalization(self, arr):
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True)
        return (arr - mean) / (std + 1e-8)  

    def __getitem__(self, index):
        vid_id = self.names[index]
        feat_arr = self.all_feat[vid_id]

        image = np.diff(feat_arr,axis = 0)
        image = np.float32(image)
        name_pair = self.pair_names[index].split('-')
        image_pairs = []

        for i in range(min(self.num_pair, len(name_pair))):
            # print(name_pair[i])
            if(name_pair==['']): 
                image_pairs.append(image)
                image_pair = image
                continue
    
            feat_arr = self.all_feat[name_pair[i]]

            image_pair = np.diff(feat_arr,axis = 0)
            # image_pair = self.z_score_normalization(feat_arr)
            # image_pair = feat_arr
            image_pair = np.float32(image_pair)
            
            image_pairs.append(image_pair)

        if len(name_pair)<self.num_pair:
            for i in range(self.num_pair - len(name_pair)):
                image_pairs.append(image_pair)

        label = torch.Tensor([float(self.labels[index])]).squeeze()
        return image, image_pairs, label
    
    def __len__(self):
        return len(self.names_labels)