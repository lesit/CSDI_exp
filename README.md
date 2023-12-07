Improvement Experiments on DDPM-based Diffusion Model for Time Series

# Overview
This project focuses on enhancing diffusion models for time series data, specifically improving the Conditional Score-based Diffusion Model (CSDI). Our goal was to test the potential of improving performance or speed by applying methods from other papers to CSDI and to deepen our understanding of diffusion models.

# Key Contributions
- Exploration of noise variation (Simplex noise vs. Gaussian noise) in diffusion models.
- Implementation of an ODE solver for handling long sampling distances, a known limitation in DDPM.
- Comparative analysis of DDPM and DDIM methodologies for efficiency and performance.
- Development and analysis of enhanced models using a healthcare dataset from the PhysioNet Challenge 2012.

# Repository Contents
- Experimental code for model development and analysis.
- Data preprocessing and analysis scripts.
- Documentation of methodologies and results.

# Future Work
- We plan to extend our research to generative models for time series data and explore reinforcement learning using data augmented by these generative models.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
