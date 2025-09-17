# TrED Evaluation Framework

This repository provides the official implementation of the Evaluation Framework for Transfer Learning in Evolving Domains (TrED).
It has been developed to support research and applied work in scenarios where data from multiple domains evolves over time, a common challenge in real-world applications such as fraud detection.

The framework allows researchers and practitioners to:
- load and/or sample datasets from different domains, to study many transfer learning scenarios;
- apply controlled transformations to the data, to mimic realistic shifts over time and across domains;
- simulate the gradual arrival of both data and labels over time.


## Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Run an experiment:
```bash
python src/tred/experiments/evaluation_over_time1.py -c experiments/config.yaml -d cpu -r test_run --trial_start 0 --trial_end 4
```


## Repo structure
```
|
|-- datasets/          # Raw dataset folders
|
|-- experiments/       # Configurations and results from experiments
|
|-- notebooks/         # Jupyter notebooks for exploration and examples
|
|-- src/tred/          # Source code for the framework
| |-- datasets/        # Dataset loaders
| |-- domain_samplers/ # Domain samplers
| |-- experiments/     # Experiment executables and scheduler logic
| |-- methods/         # Implementations of transfer learning methods
| |-- models/          # Base models used by TL methods
| |-- transformations/ # Dataset transformations
| |-- utils/           # Utility functions and helpers
|
|-- README.md          # Project documentation
|-- requirements.txt   # Python dependencies
```



## Extending the Framework

The framework was designed to be modular and easily extensible, to meet the requirements of various use cases. You can add:
- new datasets by extending `src/tred/datasets/`;
- new methods by extending `src/tred/methods/`;
- new experimental setups by extending `src/tred/experiments/`.


## Citation

If you use this framework in your work, please cite:
```
@inproceedings{
  title = {Evaluating Transfer Learning Methods on Real-World Data Streams: A Case Study in Financial Fraud Detection},
  author = {Ricardo Ribeiro Pereira, Jacopo Bono, Hugo Ferreira, Pedro Ribeiro, Carlos Soares, Pedro Bizarro},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year = {2025},
}
```
