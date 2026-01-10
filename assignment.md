# Parallel Programming for the AI Era  2025-2026

Parallel processing is fundamental to the current advancements in artificial intelligence. Training large language models relies on tens of thousands of GPUs over extended periods, and even smaller machine learning workflows benefit from distributed computing for efficient data processing.
This course introduces parallel programming with an emphasis on its application in AI. Topics include programming abstractions such as collective communication and task parallelism, popular tools and environments, and real-world parallelization strategies for AI workloads.

The course is organized around two themes:

- Collective communication methods used for training large neural networks
- Distributed programming in Python, using Dask and Ray, for speeding up data processing, machine learning, and reinforcement learning tasks A basic familiarity with parallel programming concepts (such as threads and MPI) is expected. No prior knowledge of neural networks is required; essential concepts will be introduced as part of the course.

## Link to this directory:

- <https://tinyurl.com/mryx828y>

## Class slides

1. [Introduction: Supercomputer Hardware](https://cloud.univ-grenoble-alpes.fr/s/nKNcqMRM7JqpYty)
2. [Task Programming with Dask, Ray and quick intro to Reinforcement Learning](https://cloud.univ-grenoble-alpes.fr/s/RS39LJfBkfmcnbG) 
3. [Collective Communications](https://cloud.univ-grenoble-alpes.fr/s/cPeo2j4cx9XzxPn)
4. [Parallel Deep Learning](https://cloud.univ-grenoble-alpes.fr/s/iZLNxKGzeg5mFHF)


## Some code

1. First  MPI run on G5K: <https://cloud.univ-grenoble-alpes.fr/s/XbPFcNfeTAk78Yi>
2. Dask: <https://cloud.univ-grenoble-alpes.fr/s/LcqYYbpqBf5cdLa>


## Homeworks and Exam

- At the end of the last class: you will be asked to take a paper&pen short exam (quiz) without access to any digital device.
- At home: Reinforcement learning experiments with RLlibs (<https://docs.ray.io/en/latest/rllib/index.html>) 
  - Can be done individually or by groups of 2 (not more)
  - You cans start early, no need to deeply understand everything to start.
  - On Grid'5000 cluster  (<https://www.grid5000.fr/>) 
    - I will create you an account. You will receive activation instructions by emails. Then fellow instructions (<https://www.grid5000.fr/w/Getting_Started>)
  - Deploy an reinforcement learning problem using RLlibs (<https://docs.ray.io/en/latest/rllib/index.html>) and some existing use case (from gymnasium)
  - More ambitious students can try to implement their own actor/critic RL algo with Ray-core and Pytorch (in that case a more basic use case and results will be ok)
  - Test performance  with a single node and then with at least 2 nodes nodes (using GPUs is faster but RL neural networks are usually quite small so it can also perfectly run on CPUs only)
  - Store code and short report in an accessible (for me)  git repo (report can ba a markdown file, a jupyter notebook.) 
    - Explain who did what if work of 2 persons.
  - Send me the link to the repo before 12/01/2026 ([bruno.raffin@inria.fr](mailto:bruno.raffin@inria.fr))
