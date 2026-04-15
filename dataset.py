import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

class JetDataset(Dataset):
    def __init__(self, h5_file, tag : str, 
                 num_phys_bins=(40, 30, 30), 
                 num_const = 50,
                 num_features = 3,
                 add_start=True, 
                 add_stop=True,
                 n_jets = None,
                 key = "discretized",
                 h5 = False
                 ):
        
        self.h5_file = h5_file
        self.num_phys_bins = num_phys_bins
        self.add_start = add_start
        self.add_stop = add_stop
        self.tag = tag

        print(f"Loading Dataset: {tag} | n_jets = {n_jets} | {h5_file}")

        if h5:
            with h5py.File(h5_file, 'r') as f:
                df = pd.DataFrame(np.array(f[key][:n_jets])) 
        else:
            df = pd.read_hdf(h5_file, key = key, stop= n_jets) 

        self.jets = self.disc_to_token(df,
                                       num_phys_bins=num_phys_bins,
                                       num_features=num_features,
                                       num_const=num_const,
                                       to_tensor=True,
                                       add_start=add_start,
                                       add_stop=add_stop)

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, index):
        return self.jets[index]

    def disc_to_token(self, df, 
                      num_phys_bins=(40, 30, 30),
                      num_features = 3,
                      num_const = 50, 
                      to_tensor = True,
                      add_start=True, 
                      add_stop=True):
        """
        Converts discretized jet data to token representation.
        Input: DataFrame with shape (num_jets, num_particles, 3) where the last dimension corresponds to (pt_bin, eta_bin, phi_bin)
        Output: Tensor of shape (num_jets, seq_len, 3) where seq_len = num_particles + 2 (if add_start and add_stop are True) and the last dimension corresponds to (pt_token, eta_token, phi_token)
        """

        x = df.to_numpy(dtype = np.int64)[:, :num_const * num_features]
        x = x.reshape(x.shape[0], -1, num_features) #this reshapes the data such that its 3 dimensional with [njets, nconst, nfeatures] [[[pt_1, eta_1, phi_1], ...],[[pt_1, eta_1, phi_1], ...],...]
    
        x = x.copy()

        padding_mask = x == -1 #marks every where a invalid const is

        #add start and stop token if needed
        if add_start: 
            #this shifts every valid bin --> 1 so the 0 can now be the start token
            x[~padding_mask] += 1
            
            #this adds a start particle with (0,0,0) to the start of every jet
            x = np.concatenate(
                (
                    np.zeros((len(x), 1, num_features), dtype=int),
                    x,
                ),
                axis=1,
            )

            num_bins= [x +1 for x in num_phys_bins]
            #print("Added start token. New bins are now:", num_bins)
        #add stop token only if the actual number of const. in the jet is smaller than the limit we have set for const.
        #so if a jet fills all the const dont set a stop token
        if add_stop:
            num_bins= [x +1 for x in num_bins]
            
            #compute length of each jet
            jet_length = (~padding_mask[:, :, 0]).sum(1) + 1 #this gives the index of the first invalid const. +1 because of start token
            valid = (jet_length >= 0) & (jet_length < x.shape[1]) #this ensures that the index we want to set to the stop token is not out of bounds 

            x[np.arange(x.shape[0])[valid], jet_length[valid]] = num_bins        

            #print("Added stop token. New bins are now:", num_bins)

        if to_tensor: 
            x = torch.tensor(x)

        return x
