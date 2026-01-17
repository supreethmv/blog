---
title: "Quantum-Powered Asset Clustering: Correlation Clustering with GCS-Q"
description: "A step-by-step guide to clustering financial assets using signed correlation graphs and quantum annealing with the GCS-Q algorithm."
author: supreeth
date: 2025-12-12 10:00:00 +0800
last_modified_at: 2026-02-17 10:00:00 +0800
categories: [Quantum Algorithms, Quantum Annealing, Machine Learning]
tags: [Unsupervised Learning, DWave, D-Wave, Quantum Annealing, Clustering, Correlation Clustering]
math: true
image:
  path: /assets/img/correlation-clustering/methodology.png
  lqip: data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAICAIAAABPmPnhAAAACXBIWXMAAAsTAAALEwEAmpwYAAAA6klEQVR4nDXJQUvDMBgA0Px22ZgnDwPFgyiooCgVD4N5sQwpxZGRtWNdWbupsTPplib5krRjuw0F3/UhpZQGbf+5ut7t9s22sc6CM0jKSikJoN1fa1ODbbRx5hcgqYTW0oB2zoGxa+lK4TaVNdpYY9FKMFbxH8GY4MWaLRhfcklL2MhG6i2aszwvFzjB/cFLz+8PcJDQj9nnKi9U9q0Q+YoIJZ7vdU6P29325f15GIfheDLK+DDlKC6SqJh4r0+tbufopHVxcxaQIBiN8Zy9zxhKy2XCM5+8XT/fXXm3j72HMMLDOJ1SmFJ7AAOnzDde4KPsAAAAAElFTkSuQmCC
  alt: Overview of the GCS-Q approach for correlation clustering of financial assets.
---

## Introduction

Portfolio managers and quantitative analysts spend a great deal of effort trying to understand how financial assets move together. If two stocks go up and down in near lockstep, holding both does little for diversification. Conversely, assets that move in opposite directions can hedge one another. The standard tool for quantifying these co-movements is the **Pearson correlation coefficient**, which yields values in $[-1, 1]$: a value near $+1$ means the two assets are highly correlated, near $-1$ means they are anticorrelated, and near $0$ means they are essentially uncorrelated.

A natural next step is to **cluster** assets into groups such that **intra-cluster correlations are high** (the assets in each group move together) and **inter-cluster correlations are low or negative** (different groups capture different market factors). This is the asset clustering problem, and it lies at the heart of:

- **Portfolio optimization** — Markowitz mean-variance optimization [[5](#ref5)] benefits from well-separated clusters, since each cluster represents a distinct risk-return profile.
- **Statistical arbitrage** — pairs or basket trading strategies rely on identifying groups of cointegrated or highly correlated assets [[6](#ref6)].
- **Risk management** — understanding correlation structures helps quantify systemic risk and contagion.

### Why Classical Methods Struggle

Classical clustering algorithms such as $k$-Means, $k$-Medoids (PAM), hierarchical clustering, and even spectral methods like SPONGE [[7](#ref7)] face four fundamental limitations when applied to signed correlation graphs:

#### (i) Number of clusters $k$ is not known in advance
Most classical methods require the user to specify $k$ before running the algorithm. In practice, the "right" number of clusters in a correlation matrix is unknown and changes over time as market regimes shift. Heuristics like the elbow method, silhouette analysis, or the eigengap criterion are expensive, data-dependent, and do not generalize across datasets.

#### (ii) Reformulation is lossy
Since classical methods require a positive-definite distance metric, Pearson correlations ($\rho \in [-1, 1]$) must be transformed into non-negative distances via mappings such as:

$$
d_{ij}^{(\alpha)} = \sqrt{\alpha(1 - \rho_{ij})}, \quad \alpha > 0
$$

While this preserves ranking, it **loses semantic fidelity**. For example, a zero correlation $\rho_{ij} = 0$ (no relationship) maps to $\sqrt{\alpha}$, falsely implying a fixed non-zero dissimilarity. The rich signed structure of the original correlation graph is irreversibly discarded.

#### (iii) Optimization objective is different
Centroid-based methods (e.g., $k$-Means, PAM) optimize a **proxy objective** — minimizing distances to cluster centroids — rather than the **true objective** of maximizing intra-cluster correlations and minimizing inter-cluster correlations. Even when the distance transformation is applied, optimally solving the transformed problem does **not** recover the optimal clustering under the original signed graph.

#### (iv) Not robust for varying-sized clusters
Spectral methods like SPONGE perform well on sparse graphs with discrete weights $\in \{-1, 0, +1\}$ and approximately equal cluster sizes, but degrade significantly on dense financial correlation graphs with continuous weights and heterogeneous cluster structures. Similarly, $k$-Means assumes spherical, equally-sized clusters — an assumption that rarely holds in practice.

### Enter GCS-Q

The **Graph-based Coalition Structure Generation algorithm (GCS-Q)** [[3](#ref3)] was originally developed for cooperative game theory but is a natural fit for signed graph clustering. It operates **directly on the signed, weighted correlation graph**, avoiding lossy distance transformations. Starting with all assets in a single cluster, it **recursively bipartitions** each subgraph by solving a minimum-cut formulated as a **QUBO (Quadratic Unconstrained Binary Optimization)** problem, which is solved using a **D-Wave quantum annealer**. The algorithm **automatically determines $k$**, terminating when no further split improves intra-cluster agreement.

This post walks through the entire pipeline, from downloading stock data to running GCS-Q on D-Wave hardware, step by step.

![Methodology](/assets/img/correlation-clustering/methodology.png)
*Overview of the GCS-Q approach for correlation clustering of financial assets [[1](#ref1)]*

---

## Problem Formulation

### From Returns to a Signed Graph

Consider a set of financial assets $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$ whose historical returns over a rolling window are denoted by $r_i(t)$ for asset $a_i$ at time $t$. The pairwise Pearson correlation coefficients are computed as:

$$
\rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_{r_i} \sigma_{r_j}}, \quad \rho_{ij} \in [-1, 1]
$$

These correlations define a **signed, weighted, undirected graph** $G = (V, E, w)$ where:
- Each vertex $v_i \in V$ corresponds to asset $a_i$.
- Each edge $(v_i, v_j) \in E$ carries the weight $w_{ij} = \rho_{ij}$.

Positive edges indicate co-movement; negative edges indicate anti-correlation. Unlike classical methods, **we retain the full signed structure** of this graph.

### The Clustering Objective

We seek a partition $\Pi = \{C_1, C_2, \dots, C_k\}$ of $V$ into disjoint clusters that **maximizes intra-cluster agreement**:

$$
\max_{\Pi} \sum_{C \in \Pi} \sum_{\substack{i,j \in C \\ i < j}} w_{ij}
$$

This objective promotes grouping assets with strong positive correlations while penalizing the inclusion of negatively correlated pairs within the same cluster. Importantly, the number of clusters $k$ is **not an input** — it is determined by the algorithm.

### The Penalty Metric

Since ground-truth clusters are unavailable for real financial data, we evaluate clustering quality using the **structural balance penalty**, which penalizes two types of violations:

1. **Negative intra-cluster edges**: negatively correlated assets placed in the same cluster.
2. **Positive inter-cluster edges**: positively correlated assets placed in different clusters.

$$
\text{Penalty}(\Pi) = \sum_{C \in \Pi} \sum_{\substack{i,j \in C \\ i < j \\ w_{ij} < 0}} |w_{ij}| + \sum_{\substack{C_a \ne C_b}} \sum_{\substack{i \in C_a, j \in C_b \\ i < j \\ w_{ij} > 0}} w_{ij}
$$

A **lower penalty means better clustering**, with zero representing a perfectly structurally balanced partition. Clusters with low penalty are internally cohesive and externally distinct, exactly the properties desired for portfolio diversification.

---

## The GCS-Q Algorithm

### How It Works

GCS-Q follows a **top-down hierarchical divisive** strategy:

1. **Start** with all $n$ assets in a single cluster (the "grand coalition").
2. **Bipartition** the current cluster by solving a QUBO that computes the minimum cut of the subgraph.
3. **Check the stopping criterion**: if the cut value is non-positive (i.e., further splitting does not improve intra-cluster agreement), keep the current cluster intact. Otherwise, enqueue both partitions for further splitting.
4. **Repeat** until no more beneficial splits exist.

> **Algorithm: GCS-Q for Correlation Clustering**
>
> **Input:** Weighted graph $G = (V, E, w)$ \
> **Output:** Clustering $\mathcal{C} = \{C_1, C_2, \dots, C_k\}$
>
> 1. $\mathcal{C} \leftarrow \emptyset$, $\text{Queue} \leftarrow \{V\}$
> 2. **While** Queue is not empty:
>    1. $S \leftarrow \text{Queue.pop}()$, $G_S \leftarrow$ subgraph induced by $S$
>    2. Solve QUBO: $\max \sum_{i,j \in S} w_{ij} \cdot \mathbb{I}[x_i = x_j]$
>    3. $C \leftarrow \{i \in S \mid x_i = 1\}$, $\bar{C} \leftarrow S \setminus C$
>    4. **If** $\text{cut}(C, \bar{C}) \leq 0$ or $C = \emptyset$ or $\bar{C} = \emptyset$:
>       - $\mathcal{C} \leftarrow \mathcal{C} \cup \{S\}$
>    5. **Else:**
>       - $\text{Queue} \leftarrow \text{Queue} \cup \{C, \bar{C}\}$
> 3. **Return** $\mathcal{C}$

### How the QUBO Is Constructed

It is important to emphasize that the **entire clustering task is not formulated as a single QUBO**. Instead, GCS-Q solves a **sequence of independent QUBO problems**, one at each iteration of the recursive algorithm. Each QUBO corresponds to the **minimum-cut problem** on the current subgraph — i.e., finding the best way to split the nodes of that subgraph into two groups.

The minimum cut on a weighted graph with $n$ nodes is itself a combinatorially hard problem. An exhaustive brute-force search would need to evaluate all $2^n$ possible binary assignments to determine which bipartition minimizes the cut. For even moderately sized subgraphs, this becomes infeasible on classical hardware — **beyond roughly $n \approx 25$ nodes, brute-force enumeration is computationally intractable**. Commercial solvers like Gurobi and CPLEX use branch-and-bound and cutting-plane heuristics to avoid full enumeration, but they too struggle on fully dense graphs (where every pair of nodes is connected), which is exactly the structure of financial correlation matrices.

Given the weighted adjacency matrix $W$ of a subgraph, the QUBO matrix $Q$ for the minimum-cut problem is:

$$
Q_{ii} = \sum_{j} W_{ij} \quad (\text{diagonal: degree of node } i)
$$

$$
Q_{ij} = -W_{ij} \quad (\text{off-diagonal: negative adjacency})
$$

Binary variables $x_i \in \{0, 1\}$ indicate which side of the cut each node belongs to. Minimizing $\mathbf{x}^T Q \mathbf{x}$ yields the minimum cut, which is then used to decide whether to split the subgraph.

This QUBO formulation is natively compatible with quantum annealers (D-Wave) and variational quantum algorithms (QAOA) [[4](#ref4)].

### Why Quantum?

Since each minimum-cut QUBO involves a solution space of $2^n$ binary assignments, and GCS-Q may solve up to $\mathcal{O}(n)$ such QUBOs (one per recursive split, as $k$ ranges over $[1, n]$), the overall classical complexity is $\mathcal{O}(n \cdot 2^n)$.

The quantum annealer addresses the hard inner loop: rather than enumerating $2^n$ candidate bipartitions sequentially, it **explores all possible assignments simultaneously** via quantum superposition and tunneling. This is particularly advantageous for financial correlation graphs, which are typically **fully dense** (most pairwise correlations are non-zero), making classical heuristics and pruning strategies less effective.

In other words, quantum annealing does not speed up the outer recursion (which is at most linear in $n$), but it tackles the **exponentially hard minimum-cut subproblem** at each step — the true computational bottleneck.

---

## Implementation Guide

### Prerequisites

- **Python 3.10+**
- **D-Wave account** with an API token (store it in `dwave-api-token.txt`)

### Clone the Repository

```bash
git clone https://github.com/supreethmv/Quantum-Asset-Clustering.git
cd Quantum-Asset-Clustering
```

### Install Dependencies

Use the provided `requirements.txt` to install all necessary libraries with tested, compatible versions:

```bash
pip install -r requirements.txt
```

Store your D-Wave API token in a file named `dwave-api-token.txt` in the project root.

---

### Step 1: Import Dependencies

We begin by importing the libraries we need. These fall into four logical groups:

```python
# --- Data & Numerics ---
# yfinance: download historical stock price data directly from Yahoo Finance
# pandas/numpy: data manipulation and linear algebra
import yfinance as yf
import pandas as pd
import numpy as np

# --- Graph & Visualization ---
# networkx: represent the correlation matrix as a weighted signed graph
# matplotlib/seaborn: plot penalty comparison charts
import networkx as nx
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- Quantum (D-Wave) ---
# DWaveSampler: interface to the physical quantum annealer
# EmbeddingComposite: automatically maps our problem graph onto the QPU topology
# BinaryQuadraticModel / dimod: construct and manage QUBO formulations
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
import dimod

# --- Linear Algebra ---
# eigh: compute eigenvalues of symmetric matrices (used for spectral gap estimation)
from scipy.linalg import eigh

import warnings
warnings.filterwarnings('ignore')
```

> If you are using the Gurobi solver instead of D-Wave, you can skip the `dwave.system` and `dimod` imports and instead add `import gurobipy as gp` (shown in Option B below).

---

### Step 2: Implement the GCS-Q Algorithm

#### 2a. Graph Construction

The first step is to convert the Pearson correlation matrix into a graph that GCS-Q can operate on. Each stock becomes a **node**, and each pairwise correlation becomes a **weighted edge**. Positive weights represent assets that tend to move together; negative weights represent assets that move in opposite directions.

This graph is the foundation for everything that follows — the QUBO, the minimum cut, and ultimately the cluster assignments.

```python
def construct_graph(adj_matrix):
    """Constructs a NetworkX graph from a signed adjacency matrix."""
    G = nx.Graph()
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Only add an edge if there is a non-zero correlation.
            # For financial data, this is almost always the case
            # (most stocks are correlated), resulting in a dense graph.
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])
    return G
```

> **Use-case context**: For a portfolio of 50 stocks, this creates a graph with 50 nodes and up to $\binom{50}{2} = 1225$ edges. In practice, nearly all pairwise correlations are non-zero, so the graph is fully connected (dense). This density is precisely what makes the minimum-cut problem hard for classical solvers.

#### 2b. QUBO Construction and Bipartitioning

This is the heart of the formulation. The QUBO encodes the **minimum-cut objective** for the current subgraph. Given the weighted adjacency matrix $W$, the QUBO matrix $Q$ is constructed, and then submitted to a solver to find the optimal bipartition.

We provide two solver options: the **D-Wave quantum annealer** (remote, requires API key) and the **Gurobi classical solver** (local, requires license).

```python
def get_qubo_matrix(W):
    """
    Computes the QUBO matrix for the minimum cut problem.
    
    Given the weighted adjacency matrix W of a subgraph:
      - Diagonal: Q[i,i] = sum of weights on all edges touching node i (its degree)
      - Off-diagonal: Q[i,j] = -W[i,j] (negate the adjacency weight)
    
    The resulting Q encodes the minimum-cut objective:
    minimizing x^T Q x over binary x finds the partition that
    cuts through the least total positive edge weight.
    """
    n = W.shape[0]
    Q = np.zeros((n, n))
    for i in range(n):
        # Diagonal entry: the weighted degree of node i.
        # This represents the "cost" of placing node i on one side of the cut.
        Q[i, i] = np.sum(W[i])
        for j in range(n):
            if i != j:
                # Off-diagonal entry: the negative of the edge weight.
                # When x_i ≠ x_j (nodes on different sides), this contributes
                # to the objective, penalizing cutting through strong positive edges.
                Q[i, j] = -W[i, j]
    return Q
```

**Why this works**: Consider the quadratic form $\mathbf{x}^T Q \mathbf{x}$ with binary variables $x_i \in \{0, 1\}$. When expanded, the terms involving $x_i (1 - x_j) w_{ij}$ accumulate the total weight of edges **crossing** the cut. Minimizing this is exactly the minimum cut — the partition that keeps strongly correlated assets together and separates anti-correlated ones.

> **In financial terms**: the QUBO is asking "what is the best way to split this group of stocks into two sub-groups such that stocks within each sub-group are as positively correlated as possible?" The quantum annealer (or Gurobi) answers this question at each step of the algorithm.

##### Option A: Solving on D-Wave Quantum Annealer

**Note:** This requires you to have an account on [D-Wave Leap](https://cloud.dwavesys.com/). Upon logging in, the D-Wave Leap dashboard is displayed — you can find your private token on the left column under **Solver API Token**. D-Wave offers limited free access to their quantum processing units (QPUs) for new users, which is sufficient to run the experiments in this tutorial. Store your token in a file named `dwave-api-token.txt` in the project root.

```python
def bipartition(graph):
    """
    Bipartitions a graph using D-Wave quantum annealing.
    Returns two partitions and the QPU access time.
    
    This function is called ONCE per recursive step of GCS-Q.
    Each call submits a separate QUBO to the quantum annealer,
    corresponding to the minimum-cut of the current subgraph.
    """
    # Base case: a single node cannot be split further
    if len(graph.nodes()) == 1:
        return [], [0], 0
    
    # Extract the adjacency matrix of the current subgraph
    # and build the QUBO matrix from it
    w = nx.adjacency_matrix(graph).todense()
    qubo = get_qubo_matrix(W=w)
    
    # Convert the numpy QUBO matrix into D-Wave's BinaryQuadraticModel format.
    # This is the standard input format for D-Wave samplers.
    bqm = BinaryQuadraticModel.from_qubo(qubo)
    
    # Connect to the D-Wave cloud service.
    # EmbeddingComposite handles the mapping from our logical QUBO variables
    # onto the physical qubit topology (Pegasus) of the Advantage system.
    # This embedding step is crucial: our problem graph is fully connected,
    # but the QPU has limited physical connectivity, so multiple physical
    # qubits may represent a single logical variable ("chain").
    sampler = EmbeddingComposite(
        DWaveSampler(
            token=open('dwave-api-token.txt', 'r').read().strip(),
            solver={'topology__type': 'pegasus'}
        )
    )
    
    # Submit the QUBO to the quantum annealer.
    # num_reads=1000 means the annealer runs 1000 independent annealing
    # cycles (~20 microseconds each) and returns the best solution found.
    # More reads improve robustness against hardware noise.
    sampleset = sampler.sample(bqm, num_reads=1000)
    
    # Track QPU access time for benchmarking (in microseconds)
    qpu_access_time = sampleset.info['timing']['qpu_access_time']
    
    # The best solution is a dictionary mapping node index → binary value.
    # Nodes assigned 1 go to partition1, nodes assigned 0 go to partition2.
    solution = sampleset.first.sample
    partition1 = [node for node in solution if solution[node] == 1]
    partition2 = [node for node in solution if solution[node] == 0]
    
    return partition1, partition2, qpu_access_time
```

> **Note**: The `num_reads=1000` parameter results in 1000 independent annealing cycles, each taking roughly 20 microseconds on the QPU. The total QPU time per call is therefore about 20 ms, but network latency to the D-Wave cloud adds additional overhead.

##### Option B: Solving with Gurobi (Classical)

If you do not wish to run the problem on the remote quantum annealer, or do not have a D-Wave API key, you can also use the **Gurobi** solver which runs locally on your machine.

```python
import gurobipy as gp
from gurobipy import GRB

