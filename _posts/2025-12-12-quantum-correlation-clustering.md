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
