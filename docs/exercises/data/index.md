---
exercise: data
ai_use: "LEMBRA DE PREENCHER"
---

# 1. Data

!!! abstract "Assignment"

    [Exercises → Data](https://insper.github.io/ann-dl/2026.2/exercises/data/){:target='_blank'}

**Data Preparation and Analysis for Neural Networks**

This activity explores how data spreads across space — its shape, direction, and density — and how that spread shapes the challenge of separating one class from another. Below, you'll find a series of synthetic datasets that illustrate this idea: as point clouds stretch, rotate, or overlap in different ways, the difficulty of classifying them shifts accordingly. Alongside generating and visualizing this data, the examples also touch on the practical side of getting it ready for a neural network, from cleaning up irregularities to formatting it for training.

## Exercise 1 - Point Clouds: Geometry and Spread in 2D

Before jumping into network architecture, it's worth understanding how the data itself is distributed. Here, I generate and measure two-dimensional point clouds, looking at how their spread and shape influence the complexity of the decision boundaries a network would need to learn to tell them apart.

### A - Generating the Clouds

Here, a synthetic dataset of 400 points is built across 4 classes, 100 points each. Every class is generated from a Gaussian distribution, with each one defined by its own set of parameters—controlling where it's centered and how it spreads.

### B - More or Less Spread Out

The standard deviation determines how tightly or loosely each class's points cluster around its center. Here, the same 4 classes are regenerated four separate times, keeping the means fixed but scaling all the standard deviations by different factors. Larger spreads make the clouds blend together, while smaller ones keep them cleanly separated—letting the effect of spread on separability be seen directly.

### C - Analysis

## Exercise 2 - Non-Linearity in Higher Dimensions

A simple network like a Perceptron can only draw straight-line boundaries between classes. Deep networks earn their advantage precisely when the data can't be split that way. This part compares two datasets of the same dimensionality, side by side, to make that contrast concrete.

### A - Dataset I: Shifted Gaussians

For this dataset, 500 samples are generated for Class A and another 500 for Class B, each drawn from a multivariate normal distribution using specified parameters.

### B - Dataset II: Concentric Shells

Here, a second dataset is built—again 500 samples per class, still in 5 dimensions—but this time with a radial structure instead of separate centers:

1. Directions are drawn uniformly over the unit sphere of $\mathbb{R}^5$: a vector $v \sim \mathcal{N}(0, I_5)$ is sampled and normalized, $u = v / \|v\|$;
2. **Class C (core)**: radius $\rho \sim \mathcal{N}(2.0, 0.4)$;
3. **Class D (shell)**: radius $\rho \sim \mathcal{N}(5.0, 0.4)$;
4. Each point is then placed at $x = \rho \cdot u$, combining its direction with its radius.

### C - Visualize and Compare

### D - Analysis