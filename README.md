# TopoART_layers

Custom MATLAB deep learning layers that expose [TopoART](https://www.libtopoart.eu/) neural networks (an Adaptive Resonance Theory variant) as drop-in heads for `dlnetwork`. The layers wrap the .NET library [LibTopoART.Compatibility](https://github.com/MarkoTscherepanow/LibTopoART.Compatibility). TopoART is well-suited for tasks that require stable incremental learning after deployment or the ability to detect inputs lying outside the training distribution.

## Why TopoART as a layer?

TopoART is **not** trained by gradient descent. It learns incrementally, sample-by-sample, building stable category prototypes (and a topology between them). The confidence it produces is a similarity-to-known-data score, not a normalised softmax probability — so it can flag inputs that lie outside the training distribution instead of extrapolating with high confidence.
