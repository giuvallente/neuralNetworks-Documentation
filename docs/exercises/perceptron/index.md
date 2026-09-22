---
exercise: perceptron
ai_use: "LEMBRA DE PREENCHER"
---

# 2. Perceptron

!!! abstract "Assignment"

    [Exercises → Perceptron](https://insper.github.io/ann-dl/2026.2/exercises/perceptron/){:target='_blank'}

## Understanding Perceptrons and Their Limitations

This section looks at Perceptrons and where their limits show up. The same perceptron is trained on two datasets - one it can solve cleanly, and one it can't - and the focus isn't just on that second failure, but on *how* it fails: what the learning process looks like when separability breaks down.

## Exercise 1 - Separable Data: The Case The Perceptron Was Designed For

### A - Generating the Data

Two classes of 2D points are generated, 1000 samples per class, drawn from multivariate normal distributions:

* **Class 0**: Mean = $[1.5, 1.5]$, Covariance = $[[0.5, 0], [0, 0.5]]$
* **Class 1**: Mean = $[5, 5]$, Covariance = $[[0.5, 0], [0, 0.5]]$

The means sit far apart relative to the spread, so the two clouds end up linearly separable, with at most a handful of exceptions.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise1.py:itemA"
```

Figure 1 shows a scatter plot of all 2000 points, colored by class.

![Figure 1](figures/figure1.png)

### B - Implementing the Perceptron

A perceptron is one of the simplest possible neural networks: a single artificial neuron that takes a set of inputs, weighs them, sums them up, and passes the result through a step function to produce a binary output. It learns by adjusting those weights whenever it gets a prediction wrong, nudging its decision boundary - always a straight line (or hyperplane, in higher dimensions) - a little closer to correctly separating the two classes. This makes it the natural starting point for classification: powerful enough to solve linearly separable problems, but, as later exercises show, unable to go beyond them.

A single-layer perceptron is written from scratch here, as a reusable class - the same implementation carries over into Exercise 2.

* **Prediction**: $\hat{y} = \text{step}(\mathbf{w} \cdot \mathbf{x} + b)$, where $\text{step}(z) = 1$ if $z \geq 0$ and $0$ otherwise.
* **Update rule**: for each sample $(\mathbf{x}, y)$, $\hat{y}$ is computed and applied as

$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}, \qquad b \leftarrow b + \eta (y - \hat{y})$$

with $y, \hat{y} \in \{0, 1\}$. The error $(y - \hat{y})$ is 0 on a correct prediction - so correctly classified samples produce no update — and $+1$ or $-1$ on the two kinds of mistake.

> **Why not $\mathbf{w} \leftarrow \mathbf{w} + \eta y \mathbf{x}$?**
> That form belongs to the convention where labels are $-1$ and $+1$. With the $0/1$ labels used here, it would never update on Class 0, so the perceptron could never correct a false positive — the rule has to match the labels.

* **Initialization**: $\mathbf{w}$ is drawn from `rng.normal(0, 0.01, size=2)`, with $b = 0$. Starting from $\mathbf{w} = \mathbf{0}$ is avoided on purpose: with the learning rate under scrutiny later on, an all-zero start makes the rate provably irrelevant, since it only rescales $\mathbf{w}$ without changing the decision boundary or the epoch count.
* **Learning rate**: $\eta = 0.01$.
* **Stopping**: training continues until a full pass over the dataset produces no update, or for a maximum of 100 epochs, whichever comes first — with accuracy on the full dataset recorded after every epoch.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise1.py:itemB"
```

### C - Training and Measuring

The model is trained on the dataset, converging after 28 epochs with a final accuracy of 1.0000, final weights $\mathbf{w} = [0.0530, 0.0246]$, and final bias $b = -0.2600$.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise1.py:itemC"
```

Figure 2 shows the decision boundary $\mathbf{w} \cdot \mathbf{x} + b = 0$ drawn over the data points, with any misclassified points marked differently - none remain, since the final accuracy is perfect.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise1.py:itemC-plot"
```

![Figure 2](figures/figure2.png)

Figure 3 plots accuracy against epoch, tracking the climb from chance level (0.50) up to full separation by epoch 27–28. The rise isn't perfectly monotonic — accuracy dips and spikes along the way (e.g. epoch 9 jumps to 0.8485 before falling back to 0.6030 at epoch 10) - reflecting how each weight update can temporarily overcorrect before the boundary settles.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise1.py:itemC-plot-accuracy"
```
![Figure 3](figures/figure3.png)

### D - Analysis

Answering the questions below:

1. Why does separable data converge quickly? Connect your answer to the update rule: what happens to the number of updates per epoch as training progresses?

    **A:** The two classes are linearly separable by construction: the means are `(1.5, 1.5)` and `(5.0, 5.0)`, a Euclidean distance of ≈4.95, while each feature has standard deviation ≈0.71 (σ² = 0.5). That puts the classes roughly 3.5σ apart, so a separating hyperplane exists with a comfortable margin. The Perceptron Convergence Theorem guarantees that if such a hyperplane exists, the algorithm finds one in a finite number of updates, bounded by `(R/γ)²`, where `R` is the maximum sample norm and `γ` is the margin of the best separator. A larger margin means a smaller bound and faster convergence.

    Connecting this to the update rule `w ← w + η(y − ŷ)x`: an update only happens when a sample is misclassified (`y − ŷ ≠ 0`). Each misclassified update rotates/shifts the decision boundary a bit closer to a fully separating position. As training progresses, early epochs still have several points on the wrong side of the boundary, so several updates occur; as the boundary approaches a valid separator, fewer points are misclassified, so updates per epoch decrease; once the boundary fully separates the data, a full pass produces zero updates and training stops. This is exactly what we observe empirically for `η = 0.01`, where the number of updates per epoch is `[2, 3, 4, 3, 4, 3, 4, 3, 4, 2, 4, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 1, 0]` — it oscillates at a low level, then drops to 1 and finally 0, triggering the stopping condition after 28 epochs.

2. Re-run the training with η = 1.0. Report the epoch count and final accuracy, and compare the direction of w (w/‖w‖) against the η = 0.01 run. Explain what η controls, given that every update adds ηx to weights that started at a magnitude of about 0.01.

    **A:** Keeping the same initial draw of `w` (≈ `[0.00305, -0.01040]`, magnitude ≈0.011) and everything else identical, only changing `η`: with `η = 0.01` training converges in 28 epochs, final accuracy 1.0000, final `w = [0.0530, 0.0246]` (‖w‖ = 0.0585), direction `w/‖w‖ = [0.907, 0.421]`. With `η = 1.0` training converges in 33 epochs, final accuracy 1.0000, final `w = [4.964, 3.767]` (‖w‖ = 6.231), direction `w/‖w‖ = [0.797, 0.605]`. Both runs reach 100% accuracy, but the final boundary direction differs: `[0.907, 0.421]` vs. `[0.797, 0.605]`.

    Why they differ: the initial `w` is not zero — it has magnitude ≈0.01, drawn in a random direction. Each update adds `ηx` to `w`, and `x` has magnitude on the order of 1–5 (the data lives around `(1.5,1.5)`–`(5,5)`). With `η = 0.01`, a single update (`0.01·x`, magnitude ≈0.01–0.05) is comparable in size to the initial random `w` (≈0.011). The random initial direction therefore has real influence for several updates: it takes many small steps for the accumulated updates to overwhelm the random starting tilt, so the trajectory of the boundary — and the order in which borderline points get corrected — is shaped partly by that initial randomness. With `η = 1.0`, a single update (magnitude ≈1–5) instantly dwarfs the ≈0.011 initial `w`. The random initialization is washed out almost immediately, and `w` is dominated by the accumulated `x` vectors from the very first update onward.

    So `η` controls the relative influence of the random initialization on the final solution, not just a uniform rescaling of the trajectory (that uniform-rescaling behavior only holds when starting from `w = 0`, see Question 3). Because the effective starting point relative to the update size differs between the two runs, the specific sequence of corrections differs, and the two runs converge to different (but both valid) separating boundaries. `η` also affects how many epochs it takes (28 vs. 33 here) — it does not change the guarantee of convergence on separable data, only the specific path and the boundary found.

3. Argue what would have happened from w = 0, b = 0. Show algebraically that running the whole training twice with η₁ and η₂ produces weights that differ only by the constant factor η₂/η₁ — so the boundary and epoch count are identical and η has no effect at all.

    **A:** Assume the same deterministic order of sample presentation in both runs, and start from `w₀ = 0`, `b₀ = 0`. Define, for a run with learning rate `η`, the accumulated unscaled update vector/scalar: `v₀ = 0, c₀ = 0`, `v_{t+1} = v_t + e_t·x_t`, `c_{t+1} = c_t + e_t`, where `e_t = y_t − ŷ_t` is the error at update step `t`. Claim: for any `η > 0`, `w_t = η·v_t` and `b_t = η·c_t`.

    Proof by induction. Base case: `w₀ = 0 = η·0 = η·v₀` (same for `b₀`). Inductive step: assume `w_t = η v_t`, `b_t = η c_t`. The prediction at step `t` is `ŷ_t = step(w_t·x_t + b_t) = step(η(v_t·x_t + c_t))`. Since `η > 0`, multiplying by `η` never changes the sign, so `step(η·z) = step(z)` for `z ≠ 0`. Hence `ŷ_t = step(v_t·x_t + c_t)` — independent of `η`. This means the error `e_t = y_t − ŷ_t` is identical regardless of which `η` is used, so the update is `w_{t+1} = w_t + η e_t x_t = η v_t + η e_t x_t = η(v_t + e_t x_t) = η v_{t+1}`, and likewise `b_{t+1} = η c_{t+1}`. This closes the induction.

    Consequence 1 — weights differ only by a constant factor: for two runs with `η₁` and `η₂` (same order of samples, both starting at zero), `w_t^{(η₁)} = η₁ v_t`, `w_t^{(η₂)} = η₂ v_t`, so `w_t^{(η₂)} = (η₂/η₁)·w_t^{(η₁)}` (and the same for `b_t`).

    Consequence 2 — identical decision boundary: the boundary is the set of points where `w_t·x + b_t = 0`. Substituting `w_t = η v_t`, `b_t = η c_t` and dividing by `η ≠ 0` gives `v_t·x + c_t = 0`, which does not depend on `η` at all. So both runs describe exactly the same boundary at every step `t`, just represented by weight vectors of different magnitude.

    Consequence 3 — identical epoch count: since `ŷ_t` (and therefore `e_t`) is identical at every step regardless of `η`, the two runs make exactly the same predictions, the same mistakes, and the same updates at every corresponding step. The stopping condition (a full epoch with zero updates) is reached at the same point in the sample sequence for both runs. Hence the epoch count is identical.

    Why this fails once `w₀ ≠ 0`: the whole argument rests on `w_t` being an exact scalar multiple of `v_t`, which in turn relies on `w₀ = η·v₀ = η·0 = 0` for any `η`. If `w₀` is instead a fixed nonzero vector, then `w_t = w₀ + η v_t` is an affine, not a linear, function of `η` — the constant `w₀` does not scale with `η`, so `step(w_t·x_t+b_t)` is no longer invariant under a change of `η`, predictions can differ between runs, and the two trajectories diverge. This is exactly the mechanism behind the different boundaries observed in Question 2, and it is precisely why item B forbids initializing `w = 0`: doing so would make `η` a cosmetic hyperparameter that quietly does nothing (as proven above), hiding the fact that it matters once initialization is random.

## Exercise 2 - Overlapping Data: The Case The Perceptron Cannot Solve

### A - Generating the Data

Two classes of 2D points are generated, 1000 samples per class:

* **Class 0**: Mean = $[3, 3]$, Covariance = $[[1.5, 0], [0, 1.5]]$
* **Class 1**: Mean = $[4, 4]$, Covariance = $[[1.5, 0], [0, 1.5]]$

The means are now close together and the spread is three times larger, so the clouds overlap heavily - no straight line separates them.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise2.py:itemA"
```

Figure 4 shows a scatter plot of all 2000 points, colored by class.

![Figure 4](figures/figure4.png)

### B - Training, Keeping the Best Weights

The implementation from Exercise 1 is reused, with the same $\eta = 0.01$ and the same 100-epoch cap - modified with a `fit` method that adds the pocket-tracking logic on top of the original update loop. Because this data isn't separable, the training loop never settles - it keeps updating right up to the epoch limit - so two sets of weights are tracked instead of one:

* the final weights — whatever the loop happens to hold after the last epoch;
* the pocket weights — the best-so-far: every time an update produces a higher accuracy on the full dataset than any previously seen, $(\mathbf{w}, b)$ is copied into the "pocket" and kept. This is the *pocket algorithm*, and the copy is the only addition to the loop.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise2.py:itemB"
```

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise2.py:itemB-weights"
```

The two sets of weights, and their accuracy, turn out as follows:

**Final weights**: $\mathbf{w} = [0.0413, 0.0415]$, $b = -0.0400$, accuracy = 0.5005

**Pocket weights**: $\mathbf{w} = [0.0064, 0.0055]$, $b = -0.0400$, accuracy = 0.7285

The two numbers end up far apart - the final-iterate accuracy lands right at 50%, no better than random guessing, precisely because the loop never converges and drifts into a bad state right on its last update. The pocket weights, by contrast, hold onto the best boundary found at any point during training - still far from perfect on data this overlapped, but noticeably better than chance. That gap is the expected outcome of training on inseparable data.

### C - Figures

Figure 5 shows both decision boundaries - final and pocket - drawn over the data points, with misclassified points marked.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise2.py:itemC-plot"
```

![Figure 5](figures/figure5.png)

Figure 6 plots two curves against epoch: the accuracy of the current weights at each step, and the best-so-far accuracy.

``` { .python .copy .select linenums="1" }
   --8<-- "exercises/perceptron/exercise2.py:itemC-plot-accuracy"
```

![Figure 6](figures/figure6.png)

### D - Analysis

1. The best straight line for this data scores about 73%. Your pocket weights should land close to that; your final weights should not. Explain the gap. Where does the final boundary sit relative to the data cloud, and why does the training loop leave it there? Hint: compare how far b moves per mistake against how far w moves, given that ‖x‖ ≈ 5 for this data.

    **A:** The two classes are Gaussians with the same covariance (σ² = 1.5 on each axis) and means one unit apart along the diagonal. Their Mahalanobis distance is `d = ‖μ₁−μ₀‖ / σ ≈ √2 / √1.5 ≈ 1.15`. For equal-covariance Gaussians, the Bayes-optimal boundary is linear, and its error rate is `Φ(−d/2) ≈ Φ(−0.58) ≈ 0.28`, i.e. ≈72–73% accuracy. No straight line can do better than this — the classes genuinely overlap in feature space, so some fraction of points will always sit on the "wrong" side of any line.

    Pocket gets close to this ceiling because it keeps the single best `(w, b)` seen at any point across the whole run. Since the perceptron update never stops nudging the boundary (there is no zero-error state to settle into), it keeps sweeping through many different boundary positions and orientations over 100 epochs. By pure exploration, some of those updates land the boundary very close to the optimal separating line — and pocket freezes that moment.

    Final does not get close, because the final weights are just whatever `(w, b)` happen to be sitting in memory when the epoch loop hits its cap — there is no reason for that particular moment to be a good one. Look at the mechanics of why it lands somewhere so poor: the update to `b` is `Δb = η·(y − ŷ) = ±η = ±0.01` — a fixed-size step every time there's a mistake, regardless of where the point is. The update to `w` is `Δw = η·(y − ŷ)·x`, and here `‖x‖ ≈ 5`, so `‖Δw‖ ≈ η·‖x‖ ≈ 0.05` — five times larger per mistake than the step on `b`. Because mistakes happen on both classes roughly equally often in this heavily overlapping region, and the two classes sit in similar parts of feature space, the `+x` corrections (from false negatives) and `−x` corrections (from false positives) largely cancel each other out over many updates — `w` random-walks without ever committing to a net direction, and so does `b`, but `b`'s steps are so much smaller that it barely moves from its initial value at all (it stays pinned near 0, since `b₀ = 0`).

    In the run tested, final `w ≈ [0.041, 0.041]` (‖w‖ ≈ 0.059) and final `b ≈ -0.040`, giving a boundary at signed distance `|b|/‖w‖ ≈ 0.68` from the origin — but the entire data cloud is centered around `(3.5, 3.5)`, at distance ≈4.9 from the origin. The final boundary sits almost at the origin, nowhere near the actual data. With the line that far from the cloud, essentially the whole cloud falls on one side of it, so the "classifier" just predicts one single class for (almost) everyone — which on a balanced dataset gives accuracy ≈50%, exactly what we observe (final accuracy ≈0.50 vs. pocket accuracy ≈0.73). The training loop leaves it there because there is no stopping condition tied to boundary quality: it simply reports the state at epoch 100, and nothing in the update rule pulls the boundary toward the data cloud once `b` gets stuck near zero.

2. Compare Figure 3 with Figure 6. In Exercise 1 the accuracy curve settles; here it does not. What does the perceptron convergence theorem guarantee, and which of its assumptions does this dataset violate?

    **A:** The Perceptron Convergence Theorem guarantees that if the training data is linearly separable — i.e., there exists some `(w*, b*)` that classifies every point correctly, with a positive margin `γ > 0` between the boundary and the closest point of either class — then the perceptron update rule makes at most a finite number of mistakes, bounded by roughly `(R/γ)²` (`R` = max sample norm). Once that bound of mistakes is exhausted, a full epoch with zero updates must occur, and training stops with 100% accuracy. That is exactly what Figure 3 shows: the accuracy curve rises and then settles at 1.0 once the finite mistake budget is used up.

    This dataset violates the separability assumption itself: with the two Gaussians overlapping (Mahalanobis distance ≈1.15, so the clouds interpenetrate), there is no `(w, b)` that classifies all 2000 points correctly — the best possible linear boundary already misclassifies ≈27% of points (as shown in question 1). Since no zero-error solution exists, there is no finite mistake bound to exhaust: the algorithm can be shown a misclassified point every single epoch, forever. That is why the current-weights curve in Figure 6 never settles — it keeps oscillating for all 100 epochs (current accuracy hovers around 0.50 near the end, still changing epoch to epoch), because the theorem's guarantee simply doesn't apply here.

3. Does adding more epochs fix it? Does a smaller η? Justify your answer from the update rule rather than by trial and error.

    **A:** No to both, and for the same structural reason. The convergence theorem's finite-mistake bound `(R/γ)²` only exists because a separating margin `γ > 0` exists. Here, `γ` doesn't exist (no line achieves zero error), so there is no finite bound to exhaust in the first place — not "a very large bound," but no bound at all. Running more epochs doesn't get you closer to exhausting something that was never finite; it just gives the random walk more time to keep wandering. The pocket accuracy plateaus once it happens to hit a near-optimal boundary (which tends to happen reasonably early, since the search space of 2D lines is small), but the current-weights curve has no reason to ever stabilize, at epoch 100 or epoch 10,000 — the update rule keeps firing an update every time a mistake shows up, and by the overlap argument in question 1, mistakes are guaranteed forever.

    A smaller `η` doesn't fix it either, for a reason visible directly in the update rule: `Δw = η·(y−ŷ)·x` and `Δb = η·(y−ŷ)`. Shrinking `η` uniformly rescales the size of every step, but it does not change the sign of `w·x+b` for any given `(w,b)` direction, and it does not change which points are on which side of a given line. In other words, `η` controls how fast the boundary moves through parameter space, not whether a mistake-free destination exists in that space. Since the geometric fact "no line separates these classes" doesn't depend on `η` at all, a smaller `η` just produces a slower version of the same endless cycling — smaller, more numerous steps retracing a qualitatively identical non-convergent path, not a path that eventually stops.

    Conclusion: neither knob addresses the root cause — the data violates the separability assumption the theorem needs — so neither can be fixed by tuning training budget or step size. The only thing that yields a useful, stable answer despite the non-separability is exactly the pocket mechanism: instead of hoping the loop settles somewhere good, it explicitly remembers the best boundary found along the way.

## Results Summary

| # | Quantity | Value |
|---|----------|-------|
| 1 | Exercise 1 — final $\mathbf{w}$ and $b$ | [0.0530, 0.0246], -0.26 |
| 2 | Exercise 1 — epochs to convergence | 28 |
| 3 | Exercise 1 — final accuracy | 1.00 |
| 4 | Exercise 1 — epochs and final accuracy with $\eta = 1.0$ | 33, 1.00 |
| 5 | Exercise 2 — final $\mathbf{w}$ and $b$ | [0.0413, 0.0414], -0.04 |
| 6 | Exercise 2 — accuracy of the final weights | 0.5005 |
| 7 | Exercise 2 — accuracy of the pocket weights | 0.7285 |
| 8 | Exercise 2 — epoch at which the pocket best occurred | 53 |