import torch
import clip
import pickle

from pathlib import Path
MODULE_DIR = Path(__file__).parent
MODEL_FILE = MODULE_DIR/'model.torch'
TAG_FILE = MODULE_DIR/'tags'

taglist = TAG_FILE.read_text().split('\n')[:-1]

class DeepDerpi:
    def __init__(self, modules):
        self.devices, self.shared = modules.devices, modules.shared
        _model, self.preprocess = clip.load("ViT-L/14@336px", device=self.devices.device_interrogate)
        self.devices.torch_gc()
        with open(MODEL_FILE, 'rb') as f:
            torch.load = modules.safe.unsafe_torch_load
            model = pickle.load(f)
            torch.load = modules.safe.load
            # sorry if you have low vram lol
            # if not self.shared.cmd_opts.no_half: model = model.half()
            if not self.shared.opts.interrogate_keep_models_in_memory: model = model.to(self.devices.cpu)
            self.model = model
            self.dtype = next(model.parameters()).dtype
    def load(self):
        self.model = self.model.to(self.devices.device_interrogate)
    def unload(self):
        if not self.shared.opts.interrogate_keep_models_in_memory:
            self.model = self.model.to(self.devices.cpu)
            self.devices.torch_gc()
    def predict(self, im) -> str:
        o = self.model(
            # .type(self.dtype).to()
            torch.unsqueeze(self.preprocess(im),0).to(self.devices.device_interrogate)
        )[0]

        return ','.join([
            taglist[i] for i in sorted(
                range(len(taglist)), key=lambda x:o[x], reverse=True
            )[:20]
        ])


