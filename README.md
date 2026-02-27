# LayerTracer
Tracing Visual Attention Maps across layers in CNNs through combining Early Exit Neural Networks and Pertubation Saliency Based Methods.

## Background
Early Exit Neural Networks are a class of Neural Networks which attach Internal Classifiers (ICs) to internal layers of a Neural Network. During inference we take predictions at each IC and based on some condition we allow an input to at earlier layers effectively reducing computatinal cost. While these methods are useful for computation efficiency they give something add an extra mode of analysis. Through ICs we are able to analyze the development of features across layers. 

### E^2CM
One specific method call E^2CM (https://ieeexplore.ieee.org/document/9891952) is one of the easiest and most flexible early exit strategies. After training a backbone CNN we can extract the mean features across classes. Then for calssification we are able to compare the class means with the current features and classifiy based on distance. This method allows for early exit with no extra training or architectural changes which is why I choose it for this project

### RISE
Pertubation Based Saliency Methods are useful for determining which regions of an image are imporant for classificaition. These methods work by removing a region of an image, observing how the prediction changes, then repeating this with a large number of random masks. Through these observation we can build a saliency map by assigining the change in prediction to the removed portion of the image. One such method is called Randomized Input Sampling for Explanation RISE (https://arxiv.org/pdf/1806.07421.pdf). While I have somewhat simplfied the implementation the basic principle remains the same.

## Method
Using these two methods I have created a framework for observing what a model is "focusing on" layer by layer, allowing visualization of the progressive refinement of visual attention in the model. This works by applying the same masking and attribution as in RISE but across layers through comparison to the stored class means.

## Get the code running

## Some Examples
