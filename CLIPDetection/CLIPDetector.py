import torch
import numpy as np

from .detectron2.engine.defaults import DefaultPredictor
from .clip import clip


class CLIPDetector:

    def __init__(self, cfg, main_classes, additional_classes):
        self._init_from_config(cfg)

        self.main_classes_num = len(main_classes)
        self.classes = main_classes + additional_classes
        self.class_encodings = self.get_text_encodings(self.classes)

    def _init_from_config(self, cfg):
        self.box_extractor = DefaultPredictor(cfg)

        self.device = torch.device(cfg.DEVICE)
        self.model, self.preprocess = clip.load(cfg.CLIP_BACKBONE, device=self.device)
        self.model.eval()

    def detect(self, img, thr_prob=0.4):
        coords, scores, labels = self.get_boxes(img)
        scores = scores.type(torch.float16)
        if coords.numel() == 0:
            return

        anchor_features = self.get_anchor_features(img, coords)
        with torch.no_grad():
            probs = (100.0 * anchor_features @ self.class_encodings.T).softmax(dim=-1)
            clip_scores, clip_labels = torch.max(probs, dim=-1)

        generic_labels = labels == self.main_classes_num
        labels[generic_labels] = clip_labels[generic_labels]
        scores[generic_labels] = clip_scores[generic_labels]

        return coords[scores > thr_prob], labels[scores > thr_prob]

    def get_text_encodings(self, texts):
        text_tokens = torch.cat([clip.tokenize(f"this is a photo of food {text}") for text in texts]).to(self.device)
        with torch.no_grad():
            text_encodings = self.model.encode_text(text_tokens)
        text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

        return text_encodings

    def get_boxes(self, img):
        img = np.array(img)[:, :, ::-1]
        with torch.no_grad():
            preds = self.box_extractor(img)['instances']
            coords = preds.pred_boxes.tensor
            scores = preds.scores
            labels = preds.pred_classes
        coords[:, :2] = torch.floor(coords[:, :2])
        coords[:, 2:] = torch.ceil(coords[:, 2:])

        return coords, scores, labels

    def get_anchor_features(self, img, coords):
        with torch.no_grad():
            anchor_features = torch.stack([
                self.preprocess(img.crop(box.cpu().numpy())).to(self.device)
                for box in coords
            ])
            anchor_features = self.model.encode_image(anchor_features)
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

        return anchor_features
