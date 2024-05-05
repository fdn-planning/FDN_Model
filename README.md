# FDN_Model
Multi-resource Dynamic Coordinated Planning of Flexible Distribution Network

## Title: Flexible distribution network planning code for Nature Communications "Multi-resource dynamic coordinated planning of flexible distribution network"


### Citation
Rui Wang, Haoran Ji, Peng Li Hao Yu, Jinli Zhao, Liang Zhao, Yue Zhou, Jianzhong Wu, Linquan Bai, Jinyue Yan, Chengshan Wang (2024). Flexible distribution network planning code for Nature Communications "Multi-resource dynamic coordinated planning of flexible distribution network", Tianjin University. https://github.com/fdn-planning/FDN_Model

**Access Rights:** Creative Commons Attribution 4.0 International

**Access Method:** Click to email a request for this code to lip@tju.edu.cn


### Code Details
Date (year) of code becoming publicly available: 2024

Code format: .py, .npy, .txt

Estimated total storage size of dataset: Less than 100 megabytes

Number of files in dataset: 6


### Description
This provides the flexible distribution network planning code, and the Python interface scripts described in the Nature Communications Paper "Multi-resource dynamic coordinated planning of flexible distribution network".

**The software environment is as follows:**
The program is developed in Pycharm 2021.2.3 Professional Edition.
The Python interpreter is established by Anaconda Navigator 1.9.12.
In the interpreter, Python 3.7 is used, and the required configuration of the Python environment is listed in the README.txt file.

**The electrical and planning parameters of the distribution network:**
The 'data83infrochkmodcord.py' file defines the structure of a practical distribution network, and performs the topological expansion of soft open points. The file provides the basic data of the physical network, which is called by the main program.

**Flexible distribution network planning:**
The 'ChancePlanCord.py' file contains the python script used to establish the multi-resource dynamic coordinated planning model of flexible distribution network. The proposed model optimises the configuration of soft open points, photovoltaics and electric vehicle charging stations over a long-term planning period. Additionally, a probabilistic framework is established to address the source-load uncertainties in python scripts. The security risks of the distribution network are formulated by chance constraints, and the stochastic nonlinear optimisation model is effectively solved based on the modified iterative algorithm. Instructions for running the program are contained in the README.txt file.

**Source-load data:**
The sample data generated from the probability distributions of sources and loads in the paper are contained in the .npy files.

**Scenario analysis:**
To elaborate the effectiveness and economy efficiency of the multi-resource dynamic coordinated planning method of flexible distribution network, several cases are designed for comparison. The code for scenario analysis is edited and supplied respectively, and the setting and corresponding relationship are contained in the README.txt file.

### Keywords
flexible distribution network, soft open point, chance constraint, dynamic coordinated planning, uncertainty

