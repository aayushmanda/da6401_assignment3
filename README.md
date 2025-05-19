# [DA6401 Assignment 3](https://wandb.ai/sivasankar1234/DA6401/reports/Assignment-3--VmlldzoxMjM4MjYzMg)



This repository contains the implementation for **DA6401 Assignment 3**, it explores sequence-to-sequence (seq2seq) models for transliteration using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina). The primary goal is to map a Devanagari script to its corresponding word in romanized string (Latin script), leveraging RNN-based architectures and attention mechanisms. The assignment is structured to help you:

- Build and analyze vanilla seq2seq models (RNN, LSTM, GRU)
- Experiment with attention-based models
- Perform hyperparameter tuning using Weights & Biases (wandb)
- Visualize model internals and attention maps

- **Assignment Report**: [WandB Report](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment3/reports/DA6401-Assignment-3-Report--VmlldzoxMjg0NDY3Mg)
- **GitHub Repository**: [aayushmanda/da6401_assignment3](https://github.com/aayushmanda/da6401_assignment3)
- **WandB Sweep Links**:
  - [Sweep Homepage (Scratch with Parallel Plot)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment2/sweeps/letqkeos?nw=nwuserda24s016)
  - [Sweep Homepage (Scratch)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment2?nw=nwuserda24s016)
  - [Sweep Homepage (Fine-tuning ResNet50)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/resnet50-pytorch-tuning/sweeps/im2xbayo?nw=nwuserda24s016)


---
Please install the dataset before running any of the .py files
```bash
wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
tar -xvf dakshina_dataset_v1.0.tar

```

## Dataset
The project uses the [**iNaturalist 12K dataset**](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). Download and extract it as follows:

```
da6401_assignment2/
├── PartA/                    # Files for custom CNN implementation
│   ├── config.yaml           # Configuration for dataset paths and hyperparameters
│   ├── visualization.py      # Utility for plotting prediction grid
│   ├── da6401_A.ipynb        # Jupyter notebook for training and evaluation
│   ├── train.py              # Script for training the custom CNN
│   ├── model.py              # Custom CNN model definition (FlexibleCNN)
│
├── PartB/                    # Files for ResNet50 fine-tuning
│   ├── config.yaml           # Configuration for dataset paths and hyperparameters
│   ├── train.py              # Script for fine-tuning ResNet50
│   ├── sweep.ipynb           # Notebook for hyperparameter sweeps with WandB
│   ├── model.py              # ResNet50 model setup and fine-tuning logic
│   ├── da6401_B.ipynb        # Jupyter notebook for fine-tuning and evaluation
│
├── requirements.txt          # Python dependencies for both parts
├── README.md                 # Project overview and setup instructions
```