from model import VPTCNNEncoder
from memory import SituationLoader

in_model = "data/VPT-models/foundation-model-1x.model"
in_weights = "data/VPT-models/foundation-model-1x-cnn.weights"

vpt = VPTCNNEncoder(in_model, in_weights, freeze=True)
vpt.eval()
expert_dataloader = SituationLoader(vpt, data_dir='data/MakeWaterfallDB')

expert_dataloader.load_encode_save_demos(num_demos=1000, save_dir="data/MakwWaterfallDB-CNNEncoded")