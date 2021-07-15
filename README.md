# CLIPDetection

CLIPDetector is a two neural networks, CLIP and Faster R-CNN, that allow you to detect classes of objects that weren't in the training set. Faster R-CNN generate two types of boxes: simple object boxes and generic object boxes. Simple object boxes are associated with the classes of objects that were in the training set. Generic object boxes are associated with objects that can't be classified. They are classified by CLIP.

## Usage

First you need to run 'setup.py' file.
