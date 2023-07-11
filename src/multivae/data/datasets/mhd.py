from typing import Union
from numpy import ndarray
from torch import Tensor
from .base import MultimodalBaseDataset, DatasetOutput
import os
import torch

def unstack_tensor(tensor, dim=0):
    
    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack

class MHD(MultimodalBaseDataset):
    

    def __init__(self, data_file:str , modalities: list = ['label', 'audio', 'trajectory','image']):

        self.data_file = data_file
        self.modalities = modalities
        if not os.path.exists(data_file):
                raise RuntimeError(
                    'Dataset not found. Please generate dataset and place it in the data folder.')

        self._s_data, self._i_data, self._t_data, self._a_data, self._traj_normalization, self._audio_normalization = torch.load(data_file)
        
        self.data = dict()
        if 'image' in modalities:
            self.data['image'] = self._i_data
        if 'label' in modalities:
            self.data['label'] = self._s_data
        if 'trajectory' in modalities:
            self.data['trajectory'] = self._t_data
        if 'audio' in modalities:
            self.data['audio'] = self._a_data

        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (t_data, m_data, f_data)
        """
        
        data = {s : self.data[s][index] for s in self.data}
        
        if 'audio' in data:
            # Audio modality is a 3x32x32 representation, need to unstack!
            audio = unstack_tensor(data['audio']).unsqueeze(0)
            data['audio'] = audio.permute(0, 2, 1)
            
        return DatasetOutput(data = data,
                             labels = self._s_data[index])
        


    def __len__(self):
        return len(self._s_data)

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization


