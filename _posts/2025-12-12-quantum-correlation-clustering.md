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

