# rednercounterexamplegenerator classifier

We generate counterexamples against a VGG16 image classifier with ImageNet weights, finetuned on a dataset of rendered images (via redner) of meshes from 12 ShapeNet object classes. This directory contains utilities for finetuning classifiers and benchmarking classifier performance.

Requires `torch >= 1.0.0`,  `torchvision >= 0.2.1`, and `scikit-learn >= 0.22.2.post1`.

## Training

Dataset format should be a directory containing a subdirectory per each class, in which all images of a given class reside. They are currently the ShapeNet IDs for the classes, although this is not required. See `dataset.py`.

`finetune.py` loads weights from the pre-trained PyTorch VGG16 model and inserts all except those of the last layer; the last layer is set to output the number of classes in the dataset.

Note that the finetuning script relies on setting a random seed to get consistent stratified train-val-test splits. If this behavior is undesirable those lines can be removed and the model can instead train on a single specified dataset.

To train, run `python finetune.py <path to dataset directory> <path to write model at>`.

## Inference

`inference.py` loads the weights of a finetuned model from the specified path and benchmarks the model on the specified dataset directory. It also records the predictions in a JSON file.

To benchmark a model, run `python inference.py <path to model> <path to dataset> <name of predictions JSON file>`.
