from model import VPTEncoder
from memory import SituationLoader

in_model = "data/VPT-models/foundation-model-1x.model"
in_weights = "data/VPT-models/foundation-model-1x-net.weights"

vpt = VPTEncoder(in_model, in_weights, freeze=True)
vpt.eval()
expert_dataloader = SituationLoader(vpt)

expert_dataloader.load_encode_save_demos(num_demos=800)