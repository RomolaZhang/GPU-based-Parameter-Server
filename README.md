# A GPU-based Parameter Server with Explicit Memory Management

### Summary:

We plan on implementing a parameter server approach to logistic regression based on stochastic gradient descent (SGD). We intend to take the data-parallel approach to process the data samples on a distributed set of worker nodes, utilize SIMD operations on GPU to boost up the highly parallelable workloads, and manage CPU/GPU memory explicitly to enable large model training and improve performance.

### Background:

Large-scale machine learning has been a hot area of research nowadays with its ample applications in recommendation systems, image classifications, etc. In this project, we plan to train a large logistic regression model on a dataset of business scale. As the amount of data storage and computation intensity of this task will exceed the capacity of one single computer, we plan to implement a parameter server to leverage multiple machines.

Compared with previous CPU-based parameter servers, we believe a GPU-based parameter sever can achieve better training efficiency due to its exceptional computation power. However, the limited amount of GPU memory limits the size of models that can be trained and the data movement will introduce extra overhead. Researchers of the machine learning system community have made extensive efforts to address this issue. In specific, we will implement the core design of a GPU-based parameter framework called GeePS so that we can successfully train a large model with millions to billions of parameters at a fast speed by utilizing the cooperation of multiple machines and GPU computing resources. 


### Challenge:

The problem is challenging because 

- Explicit synchronization and communication is crucial to implementing a parameter server in the GPU environment 
- Configuring a distributed computing environment requires careful design of communication protocol and network topology
- Data parallelism with good performance is hard to realize in synchronous SGD because GPU stalls are common while waiting for model parameter updates between iterations.
- Size of model parameters that could be supported by the GPU-based SGD logistic regression is restricted because of GPU's limited memory

The constraints that make the problem challenging are the individual memory spaces of CPU and GPU which require explicit data transfer, as well as GPU's limited device memory.

Our proposed solution is to take the GeePS paper's approach to handle these two challenges. For the first challenge, we intend to maintain a parameter cache in GPU and perform the data transfer between CPU and GPU in the background to hide the latency. For the second challenge, we intend to implement the smart data placement policy introduced in the paper, which swaps currently unused model data out of GPU, so that GPU doesn't have to store the whole parameter model.

### Resources:

We will use GHC machines for the initial implementation and psc for large-scale experiments. 
We intend to start from scratch and refer to the pseudocode of the GeePS paper.

We are primarily using the GeePS paper as our reference:

Henggang Cui, Hao Zhang, Gregory R. Ganger, Phillip B. Gibbons, and Eric P. Xing. 2016. GeePS: scalable deep learning on distributed GPUs with a GPU-specialized parameter server. In Proceedings of the Eleventh European Conference on Computer Systems (EuroSys '16). Association for Computing Machinery, New York, NY, USA, Article 4, 1–16. DOI:https://doi.org/10.1145/2901318.2901323


### Goals and deliverables:

Here are the targets that we aim to achieve:

75% target：
- Implement GPU-based logistic regression
- Implement a synchronous version of the algorithm
- Implement the message passing protocol on one single machine
- Apply distributed parameter servers to host the model
- Implement the memory management policy on the parameter model in GPU memory according to the algorithm

100% target:
- Configure network message passing protocol on a distributed cluster
- Implement the mechanism to swap unused data from GPU to CPU and overlap data transfer with computation so that our system could run relatively a large model with an acceptable performance by avoiding data stalls

125% target: 
- Experiment with a larger model, which poses a greater challenge to the system design of our project and memory management efficiency
- Explore different synchronization models (fully asynchronous, BSP, synchronous) and compare the accuracy and performance gain among them

For the poster session demo, we plan to include speedup and accuracy graphs on different model sizes and different steps of our implementation. 

### Platform Choice: 

We decide to choose C++ with CUDA programming for this project, because of its capacity to explicitly handle CPU/GPU memory and the familiarity we gained with in Assignment 2.  

### Schedule: What are you going to do each week to hit 100%?

We plan to achieve the following by the given dates:


| Date      | Goal Reached |
| ----------- | ----------- |
| 03/28	| Research on the CPU-based Parameter Server implementation and GPU based Parameter Server implementation & Find suitable datasets |
| 04/04 |  Implement the logistic regression algorithm in CUDA on medium size dataset and model |
| 04/11 |  Scale the GPU-based implementation to a larger dataset and model size |
| 04/18 |  Implement the GPU memory management algorithm and data placement of parameter cache in GPU memory |
| 04/25	|  Experiments and profiling our GPU-based parameter server and benchmark with other ML frameworks if possible |
| 04/29	|  Summarize our results and complete the report |
| 05/05	|  Prepare the presentation |