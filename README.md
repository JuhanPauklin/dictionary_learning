This repository is for the bachelor's thesis of Juhan Pauklin and is largely based on the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) repository developed by Samuel Marks, Adam Karvonen, and Aaron Mueller, which is for doing dictionary learning via sparse autoencoders on neural network activations.

The files of the original dictionary_learning repository were modified (permitted by the MIT license) to adjust it to the needs of the work done in the thesis. Namely the files training.py, buffer.py and utils.py were modified. 

As part of the thesis sparse autoencoders (SAE) were trained on the GPT-2 based model of the University of Tartu's Research Group of Health Informatics. Four jupyter notebook files were respectively used to train the SAEs, evaluate them, "run" SAEs and perform dictionary learning, capturing the features of the language model and then analyse the extracted features.

The extracted feature activations (in the form of .h5 files) are not included in the repository as the size of one file is measured in dozens of gigabytes.
