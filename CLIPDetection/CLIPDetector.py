import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from .detectron2.engine.defaults import DefaultPredictor
from .clip import clip


class CLIPDetector:

    def __init__(self, cfg, classes):
        self._init_from_config(cfg)
        self.set_classes(classes)

    def _init_from_config(self, cfg):
        self.box_extractor = DefaultPredictor(cfg)

        self.device = torch.device(cfg.DEVICE)
        self.model = clip.load(cfg.CLIP_BACKBONE, device=self.device)
        self.preprocess = transforms.Compose([
                transforms.Resize(self.model.input_resolution.item(), interpolation=Image.BICUBIC),
                transforms.CenterCrop(self.model.input_resolution.item()),
                lambda image: image / 255.,
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        self.model.eval()
        
    def set_classes(self, classes):
        self.class_encodings = self.get_text_encodings(classes)
    
    def add_classes(self, classes):
        new_classes = self.get_text_encodings(classes)
        self.class_encodings = torch.cat((self.class_encodings, new_classes), 0)

    def detect(self, img, thr_prob=0.4):
        coords = self.get_boxes(img)
        if coords.numel() == 0:
            return

        tensor_img = torch.tensor(img).to(self.device)
        anchor_features = self.get_anchor_features(tensor_img, coords)
        with torch.no_grad():
            probs = (100.0 * anchor_features @ self.class_encodings.T).softmax(dim=-1)
            clip_scores, clip_labels = torch.max(probs, dim=-1)

        return coords[clip_scores > thr_prob], clip_labels[clip_scores > thr_prob]

    def get_text_encodings(self, texts):
        text_tokens = torch.cat([clip.tokenize(f"this is a photo of food {text}") for text in texts]).to(self.device)
        with torch.no_grad():
            text_encodings = self.model.encode_text(text_tokens)
        text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

        return text_encodings

    def get_boxes(self, img):
        img = img[:, :, ::-1]
        with torch.no_grad():
            preds = self.box_extractor(img)['instances']
            coords = preds.pred_boxes.tensor
        coords[:, :2] = torch.floor(coords[:, :2])
        coords[:, 2:] = torch.ceil(coords[:, 2:])

        return coords
    
    @staticmethod
    def _wd_coords(coords):
        wh_coords = torch.zeros(coords.shape)
        wh_coords[:, 0] = coords[:, 1]
        wh_coords[:, 1] = coords[:, 0]
        wh_coords[:, 2] = coords[:, 3] - coords.type(torch.int32)[:, 1]
        wh_coords[:, 3] = coords[:, 2] - coords.type(torch.int32)[:, 0]
        
        return wh_coords.type(torch.int32).cpu().tolist()

    def get_anchor_features(self, img, coords):
        coords = self._wd_coords(coords)
        with torch.no_grad():
            anchor_features = torch.stack([
                self.preprocess(transforms.functional.crop(img.permute(2, 0, 1), *box))
                for box in coords
            ])
            anchor_features = self.model.encode_image(anchor_features)
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

        return anchor_features
