%TRAINTOPOART - Train a TopoART head on top of a frozen backbone
%   TRAINTOPOART drives online (non-gradient) training of a TopoART layer
%   that sits on top of an already-trained, frozen backbone dlnetwork. It
%   mirrors the calling style of trainingOptions/trainnet as closely as
%   possible while honouring TopoART's learning semantics:
%     - The backbone is run in inference mode; its parameters are NOT
%       updated. (Frozen backbone is a hard precondition because TopoART
%       builds stable prototypes in the feature space; if that space
%       shifts, the prototypes become invalid.)
%     - The TopoART layer's wrapped .NET network is mutated in place by
%       calling its learn method per minibatch. There is no loss function
%       and no gradient computation.
%     - MaxEpochs defaults to 1. Values greater than 1 are particularly
%       useful when the layer's wrapped network has been configured with
%       Beta_sbm > 0 (so the second-best-matching neuron takes partial
%       steps that benefit from repeated presentation). With the default
%       Beta_sbm, additional epochs are largely idempotent.
%
%   PRECONDITION (input range): For TopoART-C, every element of the feature
%   tensor that backboneNet emits must lie in [0, 1]. The recommended
%   pattern is to map into an inner sub-interval such as [0.1, 0.9] so that
%   features at inference time that drift slightly outside the training
%   distribution still fall inside [0, 1] (e.g. by ending the backbone with
%   sigmoidLayer or functionLayer(@(x) 0.5 + 0.4*tanh(x))). TRAINTOPOART
%   does NOT rescale features for you: the transform must be part of the
%   network so that training and inference apply the identical mapping.
%   Hypersphere TopoART-C is less strict wrt. the scaling interval. Larger
%   intervals need to be reflected by larger values of the radial extend
%   parameter R.
%
%   Syntax
%     net = TRAINTOPOART(backboneNet, topoArtLayer, ds)
%     net = TRAINTOPOART(backboneNet, topoArtLayer, ds, Name=Value)
%
%   Input Arguments
%     backboneNet  - Frozen dlnetwork producing features for the head. Must
%                    have a single output whose channel count matches
%                    topoArtLayer.InputLen.
%     topoArtLayer - A topoARTLayerBase subclass instance (e.g., a
%                    topoARTClassificationLayer).
%     ds           - Datastore yielding (X, T) pairs. The minibatchqueue
%                    is configured via MiniBatchFormat (default
%                    {'CB', 'CB'}).
%
%   Name-Value Arguments
%     MaxEpochs       - Number of passes over the data. (default: 1)
%     MiniBatchSize   - Samples per learn call. (default: 128)
%     Shuffle         - 'every-epoch' (default), 'once', or 'never'.
%     MiniBatchFormat - Cell of format strings for X and T. (default:
%                       {'CB', 'CB'})
%     MiniBatchFcn    - Function handle that turns the cell arrays read
%                       from the datastore into tensors matching
%                       MiniBatchFormat. The default assumes tabular
%                       data with one row per sample for both X and T,
%                       and returns ('C', 'B')-ordered matrices.
%     Verbose         - Print epoch/iteration progress. (default: true)
%
%   Output Arguments
%     net - dlnetwork with the trained topoArtLayer connected to the
%           output of backboneNet, ready for prediction.
function net = trainTopoART(backboneNet, topoArtLayer, ds, options)

    arguments
        backboneNet  (1, 1) dlnetwork
        topoArtLayer (1, 1) topoARTLayerBase
        ds
        options.MaxEpochs       (1, 1) {mustBeInteger, mustBePositive} = 1
        options.MiniBatchSize   (1, 1) {mustBeInteger, mustBePositive} = 128
        options.Shuffle         (1, :) char ...
            {mustBeMember(options.Shuffle, {'every-epoch', 'once', 'never'})} = 'every-epoch'
        options.MiniBatchFormat (1, :) cell = {'CB', 'CB'}
        options.MiniBatchFcn    (1, 1) function_handle = @defaultRowsToCB
        options.Verbose         (1, 1) logical = true
    end

    % The backbone must have a single output that we can wire into the head.
    if numel(backboneNet.OutputNames) ~= 1
        error(['backboneNet must have a single output. Found %d outputs. ' ...
            'Wire the desired feature tensor through a single output layer ' ...
            'before calling trainTopoART.'], numel(backboneNet.OutputNames))
    end

    % removeLayers drops the Initialized flag even when all remaining
    % learnables still hold trained values; running initialize is a no-op
    % for parameters that are already set, so it is safe here.
    if ~backboneNet.Initialized
        backboneNet = initialize(backboneNet);
    end

    mbq = minibatchqueue(ds, 2, ...
        MiniBatchSize   = options.MiniBatchSize, ...
        MiniBatchFcn    = options.MiniBatchFcn, ...
        MiniBatchFormat = options.MiniBatchFormat);

    if strcmp(options.Shuffle, 'once')
        shuffle(mbq)
    end

    if options.Verbose
        fprintf(['trainTopoART: %d epoch(s), MiniBatchSize = %d, ' ...
            'Shuffle = %s\n'], options.MaxEpochs, options.MiniBatchSize, ...
            options.Shuffle);
    end

    iter = 0;
    for epoch = 1:options.MaxEpochs

        if strcmp(options.Shuffle, 'every-epoch')
            shuffle(mbq)
        else
            reset(mbq)
        end

        epochSamples = 0;
        while hasdata(mbq)
            iter = iter + 1;
            [X, T] = next(mbq);

            % forward pass through the frozen backbone in inference mode;
            % no dlfeval/dlgradient, so parameters are not updated and the
            % autograd tape is not built
            features = predict(backboneNet, X);

            % reshape features to sampleNum-by-InputLen rows for the
            % .NET Learn API
            featuresRaw = double(extractdata(features));
            if size(featuresRaw, 1) ~= topoArtLayer.InputLen
                error(['Backbone output channel count (%d) does not match ' ...
                    'topoArtLayer.InputLen (%d).'], ...
                    size(featuresRaw, 1), topoArtLayer.InputLen)
            end
            featuresMat = featuresRaw';

            % targets: 'CB' -> rows-are-samples; vector targets become a
            % column vector (TopoART-C class IDs), matrix targets become
            % a sampleNum-by-targetLen matrix
            targetsRaw = double(extractdata(T));
            if isrow(targetsRaw) || iscolumn(targetsRaw)
                targetsArg = targetsRaw(:);
            else
                targetsArg = targetsRaw';
            end

            % polymorphic dispatch: each subclass validates types/shapes
            topoArtLayer.learn(featuresMat, targetsArg);

            epochSamples = epochSamples + size(featuresMat, 1);
        end

        if options.Verbose
            fprintf('  epoch %d/%d done, %d samples, total iterations: %d\n', ...
                epoch, options.MaxEpochs, epochSamples, iter);
        end
    end

    % Assemble the inference-ready network: backbone -> TopoART head.
    % addLayers/connectLayers drop the Initialized flag in the same way
    % removeLayers does; the trained backbone learnables and the (state-
    % only) TopoART layer still carry their values, so re-running
    % initialize just flips the flag.
    lgraph = addLayers(backboneNet, topoArtLayer);
    net = connectLayers(lgraph, backboneNet.OutputNames{1}, topoArtLayer.Name);
    if ~net.Initialized
        net = initialize(net);
    end

end

function [Xb, Tb] = defaultRowsToCB(dataX, dataT)
%DEFAULTROWSTOCB - Default preprocess for tabular row-per-sample datastores
%   Stacks the cell arrays read from the datastore into B-row matrices and
%   transposes them to ('C', 'B') ordering expected by featureInputLayer.

    Xb = cat(1, dataX{:})';
    Tb = cat(1, dataT{:})';

end