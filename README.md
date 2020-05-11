# rednercounterexamplegenerator
Counterexamples created with Tzu-Mao Li's [Redner software package](https://github.com/BachiLi/redner).

This software package, written in PyTorch, allows for semantic attacks on an image constructed from a 3D model. More concretely, it takes an image classifier and a 3D model (in a .obj file format), renders the 3D model, and classifies it. You may attack the semantic features of the 3D model through one of 3 attributes: pose, vertex, or lighting. You can currently  use one of 3 semantic attacks to do this -- our provided options are the Fast Gradient Sign Method, Projected Gradient Descent, and the Carlini-Wagner attack. More specific function details are concretely available in the code documentation in redner_adv.py.

Our experiments were done with the freely available ShapeNetCore dataset found [here](https://www.shapenet.org/download/shapenetcore)(you will have to make an account to download it).

A tutorial, using the enclosed demo motorcycle model, is available in scripts/tutorial.py, and the model checkpoint needed to execute the tutorial can be downloaded at [the following link](https://drive.google.com/file/d/1Mjpj6MrkhnN_TXEguAdxB4IE5lyRDnL5/view). Create a directory in your cloned repository folder called torch_models/ and save the checkpoint in it. The adversarial image is saved in the directory demo_out/ created by the script.
