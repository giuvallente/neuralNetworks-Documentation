---
exercise: data
ai_use: "LEMBRA DE PREENCHER"
---

# 1. Data

!!! abstract "Assignment"

    [Exercises → Data](https://insper.github.io/ann-dl/2026.2/exercises/data/){:target='_blank'}

## Data Preparation and Analysis for Neural Networks

This activity explores how data spreads across space — its shape, direction, and density — and how that spread shapes the challenge of separating one class from another. Below, you'll find a series of synthetic datasets that illustrate this idea: as point clouds stretch, rotate, or overlap in different ways, the difficulty of classifying them shifts accordingly. Alongside generating and visualizing this data, the examples also touch on the practical side of getting it ready for a neural network, from cleaning up irregularities to formatting it for training.

## Exercise 1 - Point Clouds: Geometry and Spread in 2D

Before jumping into network architecture, it's worth understanding how the data itself is distributed. Here, I generate and measure two-dimensional point clouds, looking at how their spread and shape influence the complexity of the decision boundaries a network would need to learn to tell them apart.

### A - Generating the Clouds

Here, a synthetic dataset of 400 points is built across 4 classes, 100 points each. Every class is generated from a Gaussian distribution, with each one defined by its own set of parameters—controlling where it's centered and how it spreads.

The parameters used to generate each cloud were:

* **Class 0**: Mean = [2, 3], Standard Deviation = [0.8, 2.5]
* **Class 1**: Mean = [5, 6], Standard Deviation = [1.2, 1.9]
* **Class 2**: Mean = [8, 1], Standard Deviation = [0.9, 0.9]
* **Class 3**: Mean = [15, 4], Standard Deviation = [0.5, 2.0]

Below, you can check the code that generated these clouds.

# INSERIR O CÓDIGO AQUI

Figure 1 below shows a 2D scatter plot of all the points, colored by class, with each cloud's center marked on the plot.

# INSERIR FIGURA 1 AQUI

### B - More or Less Spread Out

The standard deviation controls how much each cloud spreads around its center. Here, the same 4 classes from before are regenerated four times over, multiplying all standard deviations by a scale factor $s$—the means stay fixed, only the spread changes:

$$s \in \{0.5,\ 1.0,\ 2.0,\ 4.0\}$$

This produces 4 separate datasets of 4 classes each. Below, you can check the code that generated these 4 datasets.

# INSERIR CÓDIGO AQUI

Figure 2 shows 4 subplots (one per value of $s$), sharing the same axis limits so the comparison is fair.

# INSERIR FIGURA 2 AQUI

For $s = 1$ only, the separation ratio is computed for each pair of classes $(i, j)$ — a measure of how distinct two clouds are relative to their size, comparing the distance between their centers to their average spread. The higher the ratio, the more cleanly separated the classes are; values below 1 suggest the clouds are close enough to overlap:

$$r_{ij} = \frac{\|\mu_i - \mu_j\|}{\bar{\sigma}_i + \bar{\sigma}_j}, \qquad \bar{\sigma}_k = \frac{\sigma_{k,x} + \sigma_{k,y}}{2}$$

There are 6 pairs in total—these are reported in a table below, along with which pair has the smallest ratio.

# INSERIR TABELA

Since the means never change, $r_{ij}$ scales with $1/s$: this makes it possible to state what the smallest $r_{ij}$ becomes at $s = 2$ without having to generate anything new—just by scaling the existing value.

Beyond that single ratio, a broader measure is used to track separability across all four scale factors: the mixing rate, defined as the fraction of points whose *nearest* class center isn't their own. This is computed purely geometrically, by comparing each point against the 4 means.

Figure 3 plots this mixing rate against $s$, making it possible to answer: at which scale factor do the clouds stop being separable by straight lines? And what happens to the smallest $r_{ij}$ at that same point?

# INSERIR FIGURA 3

# INSERIR RESPOSTA DAS PERGUNTAS

### C - Analysis

Answering the questions below:

# INSERIR RESPOSTAS

## Exercise 2 - Non-Linearity in Higher Dimensions

A simple network like a Perceptron can only draw straight-line boundaries between classes. Deep networks earn their advantage precisely when the data can't be split that way. This part compares two datasets of the same dimensionality, side by side, to make that contrast concrete.

### A - Dataset I: Shifted Gaussians

For this dataset, 500 samples are generated for Class A and another 500 for Class B, each drawn from a multivariate normal distribution using the parameters below:

* **Class A**:

  Mean vector:
  $$\mu_A = [0, 0, 0, 0, 0]$$

  Covariance matrix:
  $$\Sigma_A = \begin{pmatrix} 1.0 & 0.8 & 0.1 & 0.0 & 0.0 \\ 0.8 & 1.0 & 0.3 & 0.0 & 0.0 \\ 0.1 & 0.3 & 1.0 & 0.5 & 0.0 \\ 0.0 & 0.0 & 0.5 & 1.0 & 0.2 \\ 0.0 & 0.0 & 0.0 & 0.2 & 1.0 \end{pmatrix}$$

* **Class B**:

  Mean vector:
  $$\mu_B = [1.5, 1.5, 1.5, 1.5, 1.5]$$

  Covariance matrix:
  $$\Sigma_B = \begin{pmatrix} 1.5 & -0.7 & 0.2 & 0.0 & 0.0 \\ -0.7 & 1.5 & 0.4 & 0.0 & 0.0 \\ 0.2 & 0.4 & 1.5 & 0.6 & 0.0 \\ 0.0 & 0.0 & 0.6 & 1.5 & 0.3 \\ 0.0 & 0.0 & 0.0 & 0.3 & 1.5 \end{pmatrix}$$

# INSERIR CODIGO AQUI

### B - Dataset II: Concentric Shells

Here, a second dataset is built—again 500 samples per class, still in 5 dimensions—but this time the two classes aren't placed at different centers. Instead, they're built around the same origin, distinguished only by *how far* their points sit from it:

1. First, a random direction is picked in 5D space: a vector $v \sim \mathcal{N}(0, I_5)$ is sampled and normalized to unit length, $u = v / \|v\|$, so every direction is equally likely;
2. **Class C (the core)**: points sit close to the origin, at a distance $\rho \sim \mathcal{N}(2.0, 0.4)$;
3. **Class D (the shell)**: points sit farther out, at a distance $\rho \sim \mathcal{N}(5.0, 0.4)$;
4. Each point is then placed at $x = \rho \cdot u$—stretching the direction $u$ out to its assigned distance $\rho$.

The result is one class forming a dense ball near the center, and the other forming a shell surrounding it.

# INSERIR CÓDIGO AQUI

### C - Visualize and Compare

Since a 5D space can't be visualized directly, dimensionality reduction is needed to see what's going on.

PCA (Principal Component Analysis) is used for this: a technique that finds the directions along which the data varies the most, and projects the points onto just the top few of those directions—in this case, the top 2. This gives the best possible 2D "flattening" of the 5D data, in the sense that it preserves as much of the original spread as it can.

Here, PCA is applied to project each dataset into 2 dimensions, producing two scatter plots side by side, colored by class.

# INSERIR CÓDIGO

The result can be seen in Figure 4.

# INSERIR FIGURA 4

The explained variance of the first two components is reported for each dataset, indicating how much of the original 5D spread survives the drop to 2D—and, from that, in which dataset the 2D projection better preserves the information relevant for classification.

# INSERIR VARIÂNCIAS E RESPOSTA DATASET SHIFTED GAUSSIANS

# INSERIR CÓDIGO

For each dataset, the distance between the class centers, $\|\mu_1 - \mu_2\|$, is computed directly in the original 5D space, without relying on any projection.

# INSERIR DISTANCIAS

# INSERIR CÓDIGO

Figure 5 then shows, for each dataset, a histogram of each point's radius $\|x\|$—how far it sits from the origin—with both classes overlaid on the same axis.

# INSERIR CODIGO

# INSERIR FIGURA 5

### D - Analysis

Answering the questions below:

# INSERIR RESPOSTAS

## Exercise 3 - Preparing Real-World Data for a Neural Network

This part shifts to a real-world dataset from Kaggle. The focus here is on preprocessing it appropriately—getting it into the shape a neural network expects, specifically one using the hyperbolic tangent (`tanh`) activation function in its hidden layers.

### A - Get to Know the Data 

The dataset used here is the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset from Kaggle, using `train.csv`—the only file with labels.

# INSERIR CÓDIGO DOWNLOAD

The goal is described first: what the `Transported` column represents, and how balanced the two labels are.

# INSERIR RESPOSTA GOAL

Next, the features are listed, split into numerical (like `Age` and `RoomService`) and categorical (like `HomePlanet` and `Destination`).

# INSERIR LISTA

# INSERIR CÓDIGO

A table then reports missing values per column, both as an absolute count and as a percentage.

# INSERIR TABELA

# INSERIR CÓDIGO

Finally, for the spending columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`), the mean, median, and maximum are reported. 

# INSERIR TABELA

# INSERIR DSCOBERTAS

# INSERIR CÓDIGO

### B - Split Before Transform

The data is split into train and test sets, 80/20, stratified by the target, using a fixed seed for reproducibility.

# INSERIR RESPOSTA SOBRE SPLIT ANTES DE TRANSFORM

# INSERIR CÓDIGO

### C - Preprocess

### D - Verify and Visualize

## Results Summary

| # | Item | Value |
|---|---------|-------|
| 1 | Mixing rate at `scale = 0.5` | |
| 2 | Mixing rate at `scale = 1.0` | |
| 3 | Mixing rate at `scale = 2.0` | |
| 4 | Mixing rate at `scale = 4.0` | |
| 5 | Smallest  | |
| 6 | Variância explicada — PC1 + PC2 | |
| 7 | Raio médio — casca interna | |
| 8 | Raio médio — casca externa | |
| 9 | Amostras de treino após o split | |
| 10 | Amostras de teste após o split | |
| 11 | Colunas com valores ausentes | |
| 12 | Features após o encoding | |
| 13 | Faixa das features após o escalonamento | |

# THIS NEEDS TO BE CHECKED