# Langevin Dynamics-Based Neural Network Optimization on Real-World Data - Unique Divine

![Python 3.7+] [![License: MIT]](https://github.com/Unique-Divine/SA-Project/blob/main/LICENSE)

[Python 3.7+]: https://img.shields.io/badge/python-3.7+-blue.svg
[License: MIT]: https://img.shields.io/badge/License-MIT-yellow.svg 

This is my final project for the Applied Stochastic Analysis (APMA 4990) course at  Columbia University.

- [Project Report (PDF)](https://github.com/Unique-Divine/SA-Project/blob/main/report-Stochastic-Analysis-Project.pdf)
- Code: 
  - To run: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Unique-Divine/SA-Project/blob/main/science.ipynb) 
  - To view: [Jupyter notebook](https://github.com/Unique-Divine/SA-Project/blob/main/science.ipynb). Note, it's messy. I didn't prepare it for other people just yet. I've only collected results for my report. 

## Usage Instructions: 
Inside [optimization.py](https://github.com/Unique-Divine/SA-Project/blob/main/optimization.py), there are implementations from scratch for both the stochastic gradient Langevin dynamics (SGLD) optimzer and the preconditioned SGLD  optimizer.  
- Li, Chen, Carlson, and Carin, 2016. Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks. [[Paper link]](https://preview.tinyurl.com/25kd89a6)
- Welling and Teh, 2011. Bayesian Learning via Stochastiv Gradient Langevin Dynamics. [[Paper link]](https://bit.ly/3ngnyRA)

This repository works as a package. The results from the research report are collected using the model I implemented in [lit_modules.py](https://github.com/Unique-Divine/SA-Project/blob/main/lit_modules.py). 