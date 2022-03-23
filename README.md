# ParallelParameterServer


### Summary:

We plan on implementing a parameter server approach to logistic regression based on stochastic gradient descent (SGD). We intend to take the data-parallel approach to process the data samples on a distributed set of worker nodes, utilize SIMD operations on GPU to boost up the highly parallelable workloads, and manage CPU/GPU memory explicitly to enable large model training and improve performance.

### Background:

Large scale machine learning has been a hot area of research nowadays with its ample applications in recommendation systems, image classifications, and etc. In this project, we plan to training a large logistic regression model on a dataset of business scale. As the amount of data storage and computation intensity of this task will exceed the capacity of one single computer, we plan to implement a parameter server to leverage multiple machines.

Compared with previous CPU-based parameter servers, we believe a GPU-based parameter sever can achieve better training efficiency due to its exceptional computation power. However, the limited amount of GPU memory limites the size of model that can be trained and the data movement will introduce extra overhead. Researchers of the machine learning system community have made extensive efforts to address this issue. In specific, we will implement the core design of a GPU-based parameter framework called GeePS so that we can successfully training a large model with millions to billions of parameters at a fast speed by utilizing the cooperation of multiple machines and GPU computing resources. 


### Challenge:

The problem is challenging because 

1. Data parallelism with good performance is hard to realize in synchronous SGD because GPU stalls are common while waiting for model parameter updates between iterations.
2. Size of parameter models that could be supported by the GPU-based SGD logistic regression is restricted because of GPU's limited memory.

The constraints that makes the problem challenging are the individual memory spaces of CPU and GPU which requires explicit data transfer, as well as GPU's limited device memory.

Our proposed solution is to take the GeePS paper's approach to handle these two challenges. For the first challenge, we intend to maintain a parameter cache in GPU and perform the data transfer between CPU and GPU in the background to hide the latency. For the second challenge, we intend to implement the smart data placement policy introduced in the paper, which swaps currently unused model data out of GPU, so that GPU doesn't have to store the whole parameter model.

### Resources:

We will use GHC machines for the initial implementation and psc for large scale experiments. 
We intend to start from scratch and refer to the psuedocode of the GeePS paper.

We are primarily using the GeePS paper as our reference:

Henggang Cui, Hao Zhang, Gregory R. Ganger, Phillip B. Gibbons, and Eric P. Xing. 2016. GeePS: scalable deep learning on distributed GPUs with a GPU-specialized parameter server. In Proceedings of the Eleventh European Conference on Computer Systems (EuroSys '16). Association for Computing Machinery, New York, NY, USA, Article 4, 1–16. DOI:https://doi.org/10.1145/2901318.2901323


### Goals and deliverables:

Here are the targets that we aim to achieve:

75% target: Implement GPU-based logistic regression under the synchronous model, apply parameter server to host the model, and realize data placement of the parameter cache in GPU memory.
100% target: Besides 75% target, also implement the mechanism to swap unused data from GPU to CPU and overlap data transfer with computation so that our system could run relatively large model with acceptable performance.
125% target: Experiment with larger model and compare the performance between models.

For the poster session demo, we plan to include speedup and accuracy graphs on different model size and different steps of our implementation. 

### Platform Choice: 

We decide to choose C++ with CUDA programming for this project, because of its capacity to explicitly handle CPU/GPU memory and the familiarity we gained with in in Assignment 2.  

### Schedule: What are you going to do each week to hit 100%?

We plan to achieve the following by the given dates:

Date	Goal Reached
03/28	Research on the CPU-based Parameter Server implementation and GPU based Parameter Server implementation & Find suitable dataset
04/04   Implement the logistic regression algorithm in CUDA on medium size dataset and model
04/11   Scale the GPU-based implementation to larger dataset and model size
04/18   Implement the GPU memory management algorithm and data placement of paramaeter cache in GPU memory 
04/25	Experiments and profiling our GPU-based parameter server and benchmark with other ML framework if possible
04/29	Summarize our results and complete the report
05/05	Prepare the presentation


challenge:
1.1 limited GPU/CPU memory: GPU memory limits the size of modesl that can be trained. make scalable.
1.2 GPU-Computing structure: difficult to support data paralelism due to GPU stallls, insuficient synchronizations/consistency
2. inter-machine communication
3. Synchronization/Consistency 
4. Ensure accuracy with BSP-style execution: negative impact of data staleness on accuracy improvements far outweighs the positive benefits of reduced communication delays.






steps:
1. Original system: Caffe: (Use libraries) to decompose matrix multiplications, convoludations, and related operations into SIMD operations. 
2. Add parameter server to original system. Use explicit Clock to refresh parameters. BSP/SSP model. Parameter server state on worker nodes CPU memory. 
Obstacles:  1. Bad performance: GPU stalls. insufficient synchronization. 2. Small job size. (Has to fit the whole model in GPU memory)
3. Parameter Cache in GPU memory: perform data movement between CPU and GPU in the background, overlap with computation. Updating Parameter Cache State in memory. Prebuild index and batch operations.
4. Swap unused data from GPU to CPU. Overlap transfer with computation.




Logistic regression or matrix factorization?
GPU/CPU



Focus:
1. prebuilt indexes for gathering the parameter values for parallel updates of disperse parameters
2. Train large models by explicitly managing the memory: buffer data from much larger CPU memory
3. level of synchronization - Asynchronous execution / BSP: Different from CPU implementation, BSP converges much faster for avoiding parameter staleness
4. GPU/CPU data placement policy




Factorbird:
1. factor matrices larger than memory: eg. 200gb

2. SGD is inherently sequential -> Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent