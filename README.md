<h1 align="center">LayerTracer</h1>

<p align="center">
Tracing visual attention maps across layers in CNNs by combining
<b>Early Exit Neural Networks</b> and
<b>Perturbation-Based Saliency Methods</b>.
</p>

------------------------------------------------------------------------

<table align="center">
  <tr>
    <td align="center">
      <img src="example_visuals/cifar10_resnet56/res_0_compressed.gif" width="300"><br>
      <sub><b>Dataset:</b> CIFAR-10<br><b>Model:</b> ResNet-56</sub>
    </td>
    <td align="center">
      <img src="example_visuals/cifar10_resnet56/res_1_compressed.gif" width="300"><br>
      <sub><b>Dataset:</b> CIFAR-10<br><b>Model:</b> ResNet-56</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="example_visuals/pets_resnet50/res_0_compressed.gif" width="300"><br>
      <sub><b>Dataset:</b> Oxford-Pets<br><b>Model:</b> ResNet-50</sub>
    </td>
    <td align="center">
      <img src="example_visuals/pets_resnet50/res_1_compressed.gif" width="300"><br>
      <sub><b>Dataset:</b> Oxford-Pets<br><b>Model:</b> ResNet-50</sub>
    </td>
  </tr>
</table>

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

python main.py <args>
```

### Command Line Arguments

| Argument | Description |
|----------|------------|
| <span style="white-space:nowrap"><code>--model</code></span> | Model architecture to use. Currently only ResNet variants are supported. Additional models can be added by modifying the argument choices in <code>main.py</code>. |
| <span style="white-space:nowrap"><code>--dataset</code></span> | Dataset to evaluate. Supported datasets: <code>cifar10</code>, <code>cifar100</code>, <code>places365</code>. To add a new dataset, create a dataset class following the structure in <code>data.py</code> and add the dataset name to the choices in <code>main.py</code>. |
| <span style="white-space:nowrap"><code>--frequency</code></span> | Number of prototypes used. Additional options can be added in <code>visualizer.py</code> inside the <code>set_granularity</code> function. |
| <span style="white-space:nowrap"><code>--result_save_path</code></span> | Path where generated GIFs will be saved. Must include a filename pattern: <code>PATH/TO/DIR/&lt;save_name&gt;.gif</code>. If multiple examples are saved, identifiers will be appended automatically. |
| <span style="white-space:nowrap"><code>--prototype_save_path</code></span> | Directory where prototypes will be saved when <code>--save_prototypes</code> is enabled. |
| <span style="white-space:nowrap"><code>--load_save_path</code></span> | Loads previously saved prototypes. The program must have already been run with <code>--save_prototypes</code>, and the same path used for <span style="white-space:nowrap"><code>--prototype_save_path</code></span> should be provided here. |

## Starter Commands

### CIFAR-10 with pretrianed ResNet56
```bash
python3 main.py --model resnet56 --dataset cifar10 --frequency all --result_save_path results/cifar10/resnet56/res.gif
```
### Oxford-Pets with pretrained ResNet50
```bash
python3 main.py --model resnet50 --dataset oxford-pets --frequency all --result_save_path results/oxford_pets/resnet50/res.gif
```
