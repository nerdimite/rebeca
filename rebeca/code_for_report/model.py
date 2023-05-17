class IntraSituationCA(nn.Module):
    '''Intra-Situation Cross Attention'''
    def __init__(self, hid_dim=1024):
        super().__init__()

        # Define some constants
        self.hid_dim = hid_dim  # dimension of the input and output vectors
        d_k = 1024  # dimension of the query and key vectors
        d_v = 1024  # dimension of the value vectors
        n_heads = 4  # number of attention heads
        assert self.hid_dim % n_heads == 0  # make sure hid_dim is divisible by n_heads

        # Define linear layers for projection
        self.Wq_sit = nn.Linear(1024, self.hid_dim)  # project situation embedding to query vector
        self.Wk_key = nn.Linear(8641, d_k)  # project keyboard action one-hot vector to key vector
        self.Wv_key = nn.Linear(8641, d_v)  # project keyboard action one-hot vector to value vector
        self.Wk_cam = nn.Linear(121, d_k)  # project camera action one-hot vector to key vector
        self.Wv_cam = nn.Linear(121, d_v)  # project camera action one-hot vector to value vector
        
        self.cross_attention = nn.MultiheadAttention(self.hid_dim, n_heads)

    def forward(self, situation, actions):

        # Project input tensors to query, key and value vectors
        q_sit = self.Wq_sit(situation)  # query vector tensor of shape [1, 1, 1024]
        k_key = self.Wk_key(actions['buttons'])  # key vector tensor of shape [1, 20, 1024]
        v_key = self.Wv_key(actions['buttons'])  # value vector tensor of shape [1, 20, 1024]
        k_cam = self.Wk_cam(actions['camera'])  # key vector tensor of shape [1, 20, 1024]
        v_cam = self.Wv_cam(actions['camera'])  # value vector tensor of shape [1, 20, 1024]

        # Concatenate all the key and value vectors along the second dimension
        k_all = torch.cat([k_key, k_cam], dim=1)  # key vector tensor of shape [1, 40, 1024]
        v_all = torch.cat([v_key, v_cam], dim=1)  # value vector tensor of shape [1, 40, 1024]

        # Apply multi-head attention on the query and key-value pairs
        attn_out, _ = self.cross_attention(q_sit, k_all.transpose(0, 1), v_all.transpose(0, 1))

        return attn_out.transpose(0, 1)

class Controller(nn.Module):
    """Applies Cross Attention on the observation embedding with the situation embeddings and the next action"""

    def __init__(self, vpt_model, hid_dim=1024, device='cuda'):
        super().__init__()

        # Define some constants
        self.hid_dim = hid_dim  # dimension of the input and output vectors
        d_k = 1024  # dimension of the query and key vectors
        d_v = 1024  # dimension of the value vectors
        n_heads = 4  # number of attention heads
        assert self.hid_dim % n_heads == 0  # make sure self.hid_dim is divisible by n_heads

        self.intra_situation_ca = IntraSituationCA()

        # Define linear layers for projection
        self.Wq_obs = nn.Linear(1024, self.hid_dim)  # project observation embedding to query vector
        self.Wk_sit = nn.Linear(1024, d_k)  # project situation embedding to key vector
        self.Wv_sit = nn.Linear(1024, d_v)  # project situation embedding to value vector
        self.Wk_next_key = nn.Linear(8641, d_k)  # project next keyboard action one-hot vector to key vector
        self.Wv_next_key = nn.Linear(8641, d_v)  # project next keyboard action one-hot vector to value vector
        self.Wk_next_cam = nn.Linear(121, d_k)  # project next camera action one-hot vector to key vector
        self.Wv_next_cam = nn.Linear(121, d_v)  # project next camera action one-hot vector to value vector

        # Define multi-head attention layer
        self.cross_attention = nn.MultiheadAttention(self.hid_dim, n_heads)
        self.alpha = 1.0

        # Define output layer for concatenation or addition
        self.Wo = nn.Linear(d_v, self.hid_dim)  # project output vector to original dimension

        # Define VPT Transformer layers
        self.vpt_transformers = VPTRecurrence(**self.load_vpt_parameters(vpt_model))
        self.dummy_first = torch.from_numpy(np.array((False,))).unsqueeze(1)

    def forward(self, observation, situation, situation_actions, next_action, state_in):
        
        # Apply intra-situation cross attention
        situation = self.intra_situation_ca(situation, situation_actions)

        # Project input tensors to query, key and value vectors
        q_obs = self.Wq_obs(observation)  # query vector tensor of shape [1, 1, 1024]
        k_sit = self.Wk_sit(situation)  # key vector tensor of shape [1, 1, 1024]
        v_sit = self.Wv_sit(situation)  # value vector tensor of shape [1, 1, 1024]
        k_key = self.Wk_next_key(next_action['buttons'])  # key vector tensor of shape [1, 1, 1024]
        v_key = self.Wv_next_key(next_action['buttons'])  # value vector tensor of shape [1, 1, 1024]
        k_cam = self.Wk_next_cam(next_action['camera']) # key vector tensor of shape [1, 1, 1024]
        v_cam = self.Wv_next_cam(next_action['camera'])  # value vector tensor of shape [1, 1, 1024]

        # Concatenate all the key and value vectors along the second dimension
        key_vec = torch.cat([k_sit, k_key, k_cam], dim=1)  # key vector tensor of shape [1, 3, 1024]
        val_vec = torch.cat([v_sit, v_key, v_cam], dim=1)  # value vector tensor of shape [1, 3, 1024]

        # Apply multi-head attention on the query and key-value pairs
        out_obs, _ = self.cross_attention(q_obs, key_vec.transpose(0, 1), val_vec.transpose(0, 1))

        # Add the output vector with the original query vector
        # alpha = torch.sigmoid(self.alpha_head(observation)).squeeze()
        out_obs = q_obs + (out_obs * self.alpha)

        # Apply output layer on the output vector
        out_obs = self.Wo(out_obs)

        # Apply VPT Transformer
        latent, state_out = self.vpt_transformers(out_obs, state_in, self.dummy_first)

        return latent, state_out