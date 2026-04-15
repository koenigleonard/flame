import torch
import torch.nn as nn
import numpy as np

class JetTransformer(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 num_layers = 10,
                 num_heads = 4,
                 num_features = 3,
                 num_bin_egdes = (40, 30, 30), #--> (41, 31, 31)
                 dropout = 0.1,
                 add_start = True,
                 add_stop = True,
                 causal_mask = True,
                 output_mode = "linear",
                 positional_encoding = False):
        
            super(JetTransformer, self).__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.num_features = num_features
            self.num_bin_egdes = num_bin_egdes
            self.causal_mask = causal_mask
            self.add_start = add_start
            self.add_stop = add_stop
            self.output_mode = output_mode
            self.dropout = dropout
            self.positional_encoding = positional_encoding

            print("Initializing model")

            #always one more bin than edges
            self.num_bins = [x + 1 for x in num_bin_egdes] # --> (41, 31, 31) num of phys bins
            self.num_phys_bins = self.num_bins.copy() #number of bins for physical tokens (without start and stop tokens)

            self.PHYS_VOC_SIZE = np.prod(self.num_phys_bins)
            self.TOTAL_VOC_SIZE = self.PHYS_VOC_SIZE

            #add start and stop tokens to bin counts
            if add_start:
                self.num_bins = [x + 1 for x in self.num_bins] #---> (42, 32, 32) num of bins
            if add_stop:
                self.num_bins = [x + 1 for x in self.num_bins] #---> (43, 33, 33) num of bins
                self.TOTAL_VOC_SIZE += 1 #add one for stop token since it is included in the output layer but not in the physical vocabulary

            #define pad, stop and start bins
            self.PAD_BIN = self.num_bins
            self.START_BIN = [0 for i in range(self.num_features)]
            self.STOP_BIN = [bin -1 for bin in self.num_bins]

            self.STOP_IDX = self.PHYS_VOC_SIZE #index of stop token in the output layer --> PHYS_TOKEN (0,...., 39400), STOP_TOKEN (39401)
            self.PAD_IDX = -1

            print(f"PAD_BIN  = {self.PAD_BIN}")
            print(f"START_BIN = {self.START_BIN}")
            print(f"STOP_BIN = {self.STOP_BIN}")

            #embedding layers for each feature
            self.feature_embeddings = nn.ModuleList(
                 [
                    nn.Embedding(embedding_dim=self.hidden_dim, num_embeddings=self.num_bins[i] + 1) #+1 for PAD_BIN
                    for i in range(self.num_features)
                 ]
            )

            #add the #num_layers of TransformerEncoder Layers
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model = hidden_dim, #expected input dimension
                        nhead=num_heads,
                        dim_feedforward=hidden_dim,
                        batch_first=True,
                        norm_first=True,
                        dropout=dropout
                    )
                    for i in range(num_layers)
                ]
            )

            self.out_norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

            if output_mode == "linear":
                self.output_layer = nn.Linear(hidden_dim, self.TOTAL_VOC_SIZE) #(cannot predict START or PAD tokens, only physical and stop tokens)
            else:
                raise ValueError("Invalid output mode. Choose 'linear'")
            
            #define loss function
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        
    def sinussoidal_positional_encoding(self, batch_size, seq_len, hidden_dim, device):
        #create position indices
        position = torch.arange(seq_len, device=device).unsqueeze(1) #shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device) * (-np.log(10000.0) / hidden_dim)) #shape: (hidden_dim/2)

        pe = torch.zeros(seq_len, hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term) #apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term) #apply cos to odd indices

        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1) #shape: (batch_size, seq_len, hidden_dim)

        return pe

    def forward(self, x):
        #x shape: (batch_size, seq_len, num_features)

        batch_size, num_const, num_features = x.shape
        assert num_features == self.num_features, f"Expected {self.num_features} features, got {num_features}"

        #create padding mask to mask invalid particles (with all features in PAD_BIN)
        padding_mask = (x[:,:,0] < 0)

        #replace all -1 bins with valid bin index PAD_BIN
        for i in range(self.num_features):
            x[:,:,i] = torch.where(x[:,:,i] < 0, self.PAD_BIN[i], x[:,:,i])

        #embed each feature and sum the embeddings
        embedded = torch.zeros(batch_size, num_const, self.hidden_dim, device=x.device)
        for i in range(self.num_features):
            embedded += self.feature_embeddings[i](x[:,:,i])
        
        #add positional encoding if enabled
        if self.positional_encoding:
            pos_emb = self.sinussoidal_positional_encoding(batch_size, num_const, self.hidden_dim, x.device)
            embedded += pos_emb

        #create casual mask if needed
        attention_mask = None
        if self.causal_mask:
            #creates a triangular casual mask
            attention_mask = torch.triu(
                torch.ones(num_const, num_const, device = x.device),
                diagonal = 1
            ).bool()

        #transformer encoder layers
        for layer in self.layers:
            embedded = layer(src=embedded,
                             src_mask=attention_mask,
                             src_key_padding_mask=padding_mask,
                             is_causal=self.causal_mask)
            
        embedded = self.out_norm(embedded)
        embedded = self.dropout(embedded)

        logits_idx = self.output_layer(embedded) #shape: (batch_size, seq_len, num_bins)

        return logits_idx
    
    def loss(self, logits_idx, targets_token):
        #compute cross entropy over full dictonary size without padding

        logits_idx = logits_idx[:,:-1].reshape(-1, self.TOTAL_VOC_SIZE) #shape: (batch_size * (seq_len - 1), num_bins)

        target_idx = self.tuple_to_index(
            targets_token[:,:,0],
            targets_token[:,:,1],
            targets_token[:,:,2],
            self.num_phys_bins
        )
        target_idx = target_idx[:,1:].reshape(-1) #shape: (batch_size * (seq_len - 1))

        loss = self.criterion(logits_idx, target_idx)

        return loss

    def probability(self,
                    logits_idx,
                    targets_token,
                    logarithmic = False,
                    topk = None):
        
        
        logits_idx = logits_idx[:,:-1] #discard the last token since it has no target

        probs = torch.softmax(logits_idx, dim=-1) #shape: (batch_size, seq_len - 1, num_bins)

        # --- TOP-K FILTERING ---
        if topk is not None:
            # find topk probs at each (B,S)
            topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

            # mask: True for entries in topk
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)

            # suppress everything else
            probs = probs.masked_fill(~mask, 0.0)

        target_idx = self.tuple_to_index(
            targets_token[:,:,0],
            targets_token[:,:,1],
            targets_token[:,:,2],
            self.num_phys_bins) #--> (0,...., 39400) for physical tokens, 39401 for stop token, -1 for padding tokens
        
        #shifts the start token so that target_idx at time step t corresponds to the predicted token at time step t-1
        target_idx = target_idx[:,1:] #shape: (batch_size, seq_len - 1)
        
        padding_mask = (target_idx == self.PAD_IDX)

        target_idx = target_idx.masked_fill(padding_mask, 0) #replace padding indices with 0 to avoid indexing errors

        probs = probs.gather(dim=-1, index=target_idx.unsqueeze(-1)).squeeze(-1) #shape: (batch_size, seq_len - 1)

        probs = probs.masked_fill(padding_mask, 1.0) #set probabilities of padding tokens to 1 so they do not contribute to the final product

        if logarithmic:
            prob = torch.log(probs).sum(dim = -1)
        else:
            prob = probs.prod(dim=-1)

        return prob
        

    @torch.no_grad()
    def sample(self,
               batch_size = 100,
               max_length = 100,
               device = None,
               temperature = 1.0,
               topk = None):
        #set device
        if device is None:
            device = next(self.parameters()).device

        #initialize input with start tokens
        start = torch.tensor(self.START_BIN, device=device).unsqueeze(0).view(1,1,self.num_features)

        #create initial input tensor filled with start tokens
        x = start.repeat(batch_size, 1, 1) #shape: (batch_size, 1, num_features)

        #mark which jets are already finished
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits_idx = self.forward(x) #shape: (batch_size, seq_len, num_bins)

            next_token_logits = logits_idx[:,-1,:] / temperature #saves only the last logit vector

            probs = torch.softmax(next_token_logits, dim=-1)

            # --- TOP-K FILTERING ---
            if topk is not None:
                # find topk probs at each (B,S)
                topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

                # mask: True for entries in topk
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask.scatter_(-1, topk_idx, True)

                # suppress everything else
                probs = probs.masked_fill(~mask, 0.0)

            next_token_idx = torch.multinomial(probs, num_samples=1).squeeze(-1) #shape: (batch_size)

            next_token_idx[finished_mask] = -1 #set next token of finished jets to -1 (PAD_IDX)

            next_token_tuple = self.index_to_tuple(next_token_idx, self.num_phys_bins) #tuple of (pt, eta, phi)

            x = torch.cat([x, torch.stack(next_token_tuple, dim=-1).unsqueeze(1)], dim=1) #append next token to input

            #update finished mask if one of the features is a stop token we stop
            finished_mask |= (next_token_idx == self.STOP_BIN[0]) #if pt token is stop token, mark jet as finished 
            finished_mask |= (next_token_idx == self.STOP_BIN[1]) #if eta token is stop token, mark jet as finished
            finished_mask |= (next_token_idx == self.STOP_BIN[2]) #if phi token is stop token, mark jet as finished

            if finished_mask.all():
                break

        out = x.clone()

        #add padding if we stopped early for every jet
        if out.size(1) < max_length + 1: #+1 because we start with a start token
            padding = torch.full((batch_size, max_length - out.size(1) + 1, self.num_features), 
                                 -1, 
                                 device=device,
                                 dtype =out.dtype) #shape: (batch_size, padding_length, num_features)
            out = torch.cat([out, padding], dim=1)

        #remove start token
        if self.add_start:
            out = out[:,1:,:]

        if self.add_stop:
            #replace all tokens after the first stop token with -1 (PAD_IDX)
            stop_mask = (out[:,:,0] == self.STOP_BIN[0]) | (out[:,:,1] == self.STOP_BIN[1]) | (out[:,:,2] == self.STOP_BIN[2])
            out[stop_mask] = -1
        
        #shift bins back because start token was removed
        if self.add_start:
            padding_mask = (out == -1)

            out[~padding_mask] -= 1

        return out

    def is_real_tuple(self, pt, eta, phi):
        #check if the tuple is a real particle tuple (not a start, stop or padding token)
        is_padding = (pt == -1) | (eta == -1) | (phi == -1)
        is_start = (pt == self.START_BIN[0]) | (eta == self.START_BIN[1]) | (phi == self.START_BIN[2])
        is_stop = (pt == self.STOP_BIN[0]) | (eta == self.STOP_BIN[1]) | (phi == self.STOP_BIN[2])

        return ~(is_padding | is_start | is_stop)


    def tuple_to_index(self, pt, eta, phi, num_bins):
        """
        Map (pt, eta, phi) bin indices into single vocabulary index.

        and padding -1 to -1 in index

        index = pt + n_pt * eta + (n_pt*n_eta) * phi
        """
        #start token should never enter here make sure it does not by replacing it with a valid token (since it will be masked out later in the loss function)
        pt = torch.where(pt == self.START_BIN[0], 0, pt)
        eta = torch.where(eta == self.START_BIN[1], 0, eta)
        phi = torch.where(phi == self.START_BIN[2], 0, phi)

        #exclude all particles with pt in -1 or PAD_BIN[0] from loss calculation by setting their joint_id to -1
        padding_mask = (pt == -1) | (eta == -1) | (phi == -1)
        padding_mask |= (pt == self.PAD_BIN[0]) | (eta == self.PAD_BIN[1]) | (phi == self.PAD_BIN[2])

        #mark all particles that are stop tokens
        stop_mask = (pt == self.STOP_BIN[0]) & (eta == self.STOP_BIN[1]) & (phi == self.STOP_BIN[2])

        joint_id = (pt-1) + (eta-1) * num_bins[0] + (num_bins[0]*num_bins[1])* (phi-1) #shift bins down by 1 to account for start token

        joint_id[stop_mask] = self.STOP_IDX
        joint_id[padding_mask] = self.PAD_IDX

        return joint_id


    def index_to_tuple(self, index, num_bins):
        """
        Inverse mapping: dictionary index -> (pt, eta, phi)

        0 -> (1, 1, 1)
        39400 -> (41, 31, 31)
        39401 -> STOP_TOKEN = (42, 32, 32)
        
        -1 -> PAD_TOKEN = (43, 33, 33)

        """

        padding_mask = (index == self.PAD_IDX)

        #+1 for start token
        pt = (index % num_bins[0]) + 1
        eta = (index // num_bins[0]) % num_bins[1] + 1
        phi = index // (num_bins[0] * num_bins[1]) + 1

        pt[padding_mask] = -1
        eta[padding_mask] = -1
        phi[padding_mask] = -1

        return pt, eta, phi