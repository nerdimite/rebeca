class Retriever:
    def __init__(self, encoder_model, encoder_weights, memory_path):
        self.vpt = VPTCNNEncoder(encoder_model, encoder_weights)
        self.vpt.eval()
        self.memory = Memory()
        self.memory.load_index(memory_path)

    def encode_query(self, query_obs):
        return self.vpt(query_obs).squeeze().cpu().numpy()

    def retrieve(self, query_obs, k=1, encode_obs=True):
        if encode_obs:
            query_obs = self.encode_query(query_obs)
        results = self.memory.search(query_obs, k=k)
        
        return results[0], query_obs


class REBECA(nn.Module):
    def __init__(self, cnn_model, trf_model, cnn_weights, trf_weights, memory_path, device='auto'):
        super().__init__()

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.retriever = Retriever(cnn_model, cnn_weights, memory_path)
        self.vpt_cnn = VPTCNNEncoder(cnn_model, cnn_weights)
        self.vpt_cnn.eval()

        self.controller = Controller(trf_model)
        if trf_weights:
            self.controller.vpt_transformers.load_state_dict(torch.load(trf_weights, map_location=torch.device('cuda')))

        # Action processing
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # Action head
        self.action_head = make_action_head(action_space, self.controller.hid_dim, **PI_HEAD_KWARGS).to('cuda')

    def forward(self, obs, state_in):
        
        # extract features from observation
        obs_feats = self.vpt_cnn(obs)

        # retrieve situation from memory
        situation, _ = self.retriever.retrieve(obs_feats.to('cpu'), k=1, encode_obs=False)

        # process retrieved situations
        situation_embed, situation_actions, next_action = preprocess_situation(situation, self.device)
        
        # forward pass through controller
        latent, state_out = self.controller(obs_feats, situation_embed, situation_actions, next_action, state_in)

        # get action logits
        action_logits = self.action_head(latent)

        return action_logits, state_out