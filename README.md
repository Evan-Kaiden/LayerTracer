<h1 align="center">LayerTracer</h1>

<p align="center">
Tracing visual attention maps across layers in CNNs by combining
<b>Early Exit Neural Networks</b> and
<b>Perturbation-Based Saliency Methods</b>.
</p>

------------------------------------------------------------------------

# Background

## Early Exit Neural Networks

Early Exit Neural Networks attach **Internal Classifiers (ICs)** to
intermediate layers of a neural network.

During inference, predictions are produced at each IC. If a prediction
is confident enough, the model **exits early**, reducing computational
cost.

Beyond efficiency, early exits provide an additional analytical benefit:
they allow us to **observe how feature representations evolve across
layers**. By analyzing predictions at each IC, we can study how the
network gradually builds semantic understanding.

------------------------------------------------------------------------

# Methods

## E²CM

One early exit strategy is **E²CM (Early Exit via Class Means)**.

Instead of training additional classifiers, this method:

1.  Trains a backbone CNN normally
2.  Computes **mean feature vectors for each class**
3.  During inference, compares intermediate features to these class
    means

Classification is performed using a **distance metric** to the class
means.

### Advantages

-   No additional training
-   No architectural modifications
-   Flexible and easy to integrate

This simplicity makes E²CM ideal for experimentation with early-exit
analysis.

Reference: https://ieeexplore.ieee.org/document/9891952

------------------------------------------------------------------------

## RISE

Another key component is **RISE (Randomized Input Sampling for
Explanation)**.

RISE is a **perturbation-based saliency method** used to determine which
regions of an image influence a model's prediction.

The method works by:

1.  Generating many random masks
2.  Applying each mask to the input image
3.  Observing how the prediction changes
4.  Aggregating the results into a **saliency map**

Pixels that consistently affect predictions receive **higher attribution
values**, indicating stronger importance.

Reference: https://arxiv.org/pdf/1806.07421.pdf

------------------------------------------------------------------------

# Method

LayerTracer combines **E²CM** and **RISE** to visualize how a CNN's
attention evolves across layers.

Instead of computing saliency only at the final output, the framework:

1.  Applies the same random masking strategy used in RISE
2.  Computes attribution at **multiple intermediate layers**
3.  Uses **distance to stored class means** for predictions at each
    layer

This produces **layer-wise saliency maps**, allowing visualization of
how the network progressively refines its focus throughout the network.

------------------------------------------------------------------------

# Running the Code

``` bash
git clone <repo>
cd LayerTracer

pip install -r requirements.txt

python run_example.py
```

------------------------------------------------------------------------

# Examples

Example visualization pipeline:

    Original Image → Early Layer Attention → Mid Layer Attention → Final Layer Attention

These visualizations show how the model transitions from **low-level
texture detection** to **high-level semantic focus**.
