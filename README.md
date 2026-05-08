# TopoART_layers

Custom MATLAB deep learning layers that expose [TopoART](https://www.libtopoart.eu/) neural networks (an Adaptive Resonance Theory variant) as drop-in heads for `dlnetwork`. The layers wrap the .NET library [LibTopoART.Compatibility](https://github.com/MarkoTscherepanow/LibTopoART.Compatibility). TopoART is well-suited for tasks that require stable incremental learning after deployment or the ability to detect inputs lying outside the training distribution.

## Why TopoART as a layer?

TopoART is **not** trained by gradient descent. It learns incrementally, sample-by-sample, building stable category prototypes (and a topology between them). The confidence it produces is a similarity-to-known-data score, not a normalised softmax probability — so it can flag inputs that lie outside the training distribution instead of extrapolating with high confidence.

Because TopoART is not a deep network, it needs another model as a backbone when raw inputs are unsuitable as features. The typical workflow is:

1. Train a backbone with `trainnet` (gradient descent) using a temporary differentiable head.
2. Strip the temporary head off and **freeze** the backbone.
3. Train a TopoART head on top of the frozen backbone via `trainTopoART` (incremental, online learning — one `LEARN` call per minibatch).
4. Use the assembled `dlnetwork` for prediction.
5. The TopoART head may be trained whenever required, for instance, if new types of input or output (e.g. classes) are observed.

The layers can also be used standalone, directly behind a `featureInputLayer`, when the raw inputs already form a suitable feature space.

## Requirements

- MATLAB R2025b (or higher) + Deep Learning Toolbox
- .NET runtime: .NET Framework 4.7.2 or higher, or .NET 6.0 or higher

## Getting started

From the [src/](src/) folder, install the .NET dependencies once:

```matlab
installLibs
```

This downloads `FSharp.Core.dll`, `LibTopoART.dll`, and `LibTopoART.Compatibility.dll` into `src/lib/`. The layer constructors load the assembly on demand.

### Standalone classification example

[classify2dExample.m](src/samples/classify2dExample.m) builds a minimal `dlnetwork` consisting of a `featureInputLayer` followed by a `topoARTClassificationLayer`, trains it on two intertwined spirals (or half-moons), and plots the classified grid:

```matlab
classify2dExample           % default: 'spirals'
classify2dExample('moons')
```
The results demonstrate that (Hypersphere) TopoART-C can easily classify data with complex distributions and is able to reject unknown samples that differ from the known data.

### Backbone + TopoART-C head example

[classifyWithBackbone2dExample.m](src/samples/classifyWithBackbone2dExample.m) runs the full workflow: pretrain a small MLP backbone with `trainnet`, strip the softmax head, train a TopoART-C head on the frozen backbone via `trainTopoART`, and compare both heads on a grid:

```matlab
classifyWithBackbone2dExample
```

Here, TopoART's capability to reject unknown data is limited by the features it obtains as input.

## Important constraints

- **Input range.** TopoART requires every input to lie in `[0, 1]`. Map into an inner sub-interval such as `[0.1, 0.9]` (e.g.,`functionLayer(@(x) 0.5 + 0.4*tanh(x))` or `sigmoidLayer`) so that inputs drifting slightly outside the training distribution at inference time still fall inside `[0, 1]`. Decide the transform **before** training TopoART and apply the identical mapping at training and inference time. Do not auto-rescale on observed values. Hypersphere TopoART is less strict wrt. the scaling interval. Larger intervals need to be reflected by larger values of the radial extend parameter `R`.
- **Frozen backbone.** When training on top of a backbone, the backbone must be frozen. TopoART builds stable prototypes called categories in the feature space; if that space shifts, the categories become invalid.
- **No `trainnet` for TopoART layers.** Train them via the layer's `learn` method (standalone) or `trainTopoART` (with a backbone).

## Related information and background

- LibTopoART homepage: https://www.libtopoart.eu/
- LibTopoART (.NET): https://github.com/MarkoTscherepanow/LibTopoART
- LibTopoART.Compatibility (.NET wrapper used here):
  https://github.com/MarkoTscherepanow/LibTopoART.Compatibility
- TopoART neural networks (MATLAB package; some helpers are reused here):
  https://de.mathworks.com/matlabcentral/fileexchange/118455-topoart-neural-networks
