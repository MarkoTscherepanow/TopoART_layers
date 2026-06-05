%CLASSIFYWITHBACKBONE2DEXAMPLE - Train TopoART on top of a frozen backbone
%   ATTENTION: This function requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs (in the parent src folder)
%   must be run before CLASSIFYWITHBACKBONE2DEXAMPLE can be used.
%
%   This example demonstrates the typical end-to-end workflow with a
%   backbone:
%     1. A small backbone is trained with trainnet on a 2D dataset using a
%        temporary differentiable head (a fully connected + softmax
%        classifier). This stage uses standard gradient descent.
%     2. The differentiable head is stripped off, leaving the trained
%        feature extractor.
%     3. The (Hypersphere) TopoART-C head is constructed and trained on the
%        frozen backbone via trainTopoART. This stage uses incremental
%        TopoART learning, not gradient descent.
%     4. The assembled (backbone + TopoART-C head) network is used to
%        predict on a uniform grid; results are plotted with a confidence
%        threshold.
%
%   The two-dimensional dataset used here contains two classes distributed
%   as two interleaved half-moons.
%
%   Syntax
%     CLASSIFYWITHBACKBONE2DEXAMPLE
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling)
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling, threshTA)
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling, threshTA, threshSM)
%
%   Input Arguments
%     useScaling - Select input normalization method (scaling or tanh-based)
%       TopoART inputs must lie in [0, 1]. This switch enables to choose
%       between linear scaling (if true) and tanh-based normalization.
%       (default: true)
%     threshTA - Confidence threshold for the TopoART head
%       This confidence measures similarity to known data, so a high
%       threshold rejects grid points that fall outside the distribution of
%       the training features. (range: [0, 1]; default: 0.98 or 0.95,
%       depending on useScaling)
%     threshSM - Confidence threshold for the original head
%       Softmax confidence is just the maximum class probability and stays
%       high almost everywhere because the classifier extrapolates without
%       in-distribution awareness. Therefore, it is usually not comparable
%       to the confidence of TopoART. (range: [0, 1]; default: 0.95)
function classifyWithBackbone2dExample(useScaling, threshTA, threshSM)

    narginchk(0, 3)
    nargoutchk(0, 0)

    if nargin < 1
        useScaling = true;
    end

    if nargin < 2
        if useScaling
            threshTA = 0.98;
        else
            threshTA = 0.95;
        end
    end

    if threshTA < 0 || threshTA > 1
        error('threshTA must have a value from the interval [0, 1].')
    end

    if nargin < 3
        threshSM = 0.95;
    end

    if threshSM < 0 || threshSM > 1
        error('threshSM must have a value from the interval [0, 1].')
    end

    % raw input dimensionality
    inputLen = 2;

    % width of the backbone's ReLU hidden layers
    hiddenLen = 32;

    % backbone output dimensionality (TopoART input)
    featureLen = 8;

    % number of TopoART modules (default: 2)
    moduleNum = 2;

    % vigilance parameter of the first module
    rho_a = 0.99;

    % radial extend R required by Hypersphere TopoART-C, set according to
    % R = sqrt(inputDimension * (dataMax - dataMin)^2) / 2
    % where (dataMax - dataMin) is the per-dimension range of the actual
    % TopoART input. Both scaling and tanh-based normalization keep the
    % input within [0, 1], so dataMax - dataMin = 1 is used here.
    R = sqrt(featureLen) / 2;

    % Choose the wrapped (Hypersphere) TopoART-C network (The string is
    % resolved to a LibTopoART.Compatibility. Network enum inside the
    % constructor, after the .NET assembly has been loaded on demand.)
    netType = 'Fast_TopoART_C';

    % disable/enable prediction using the original head (fc_out + softmax)
    showSoftmaxPrediction = false;

    % Set path so that the layer and helpers can be located when the
    % example is run from the samples folder (The wrapped .NET library is
    % loaded on demand by the constructor of topoARTClassificationLayer.)
    % onCleanup restores the user's path on normal exit, error, or Ctrl+C.
    basePath = fileparts(mfilename('fullpath'));
    srcPath  = fileparts(basePath);
    oldPath  = addpath(srcPath, fullfile(srcPath, 'helpers'));
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % seed the RNG so the moon noise, sample shuffle, and backbone
    % weight init are reproducible (drop or change the seed to explore
    % init variability)
    rng(0)

    % generate and prepare the training data
    samples = create2dMoons;
    samples = samples(randperm(length(samples)), :);

    trainX = samples(:, 1:2);
    trainT = samples(:, 3);   % class IDs in {1, 2}

    if useScaling
        scaleToInner = @(x) x;  % replaced after pretraining
    else
        scaleToInner = @(x) 0.5 + 0.5 * tanh(x);
    end

    % set topology of the base network (backbone + head)
    backboneAndHead = [
        featureInputLayer(inputLen,     Name = 'input')
        fullyConnectedLayer(hiddenLen,  Name = 'fc1')
        reluLayer(                      Name = 'relu1')
        fullyConnectedLayer(hiddenLen,  Name = 'fc2')
        reluLayer(                      Name = 'relu2')
        fullyConnectedLayer(featureLen, Name = 'features')
        % satisfy TopoART input range precondition
        functionLayer(scaleToInner,     Name = 'scale_features')
        fullyConnectedLayer(2,          Name = 'fc_out')
        softmaxLayer(                   Name = 'softmax')
    ];

    pretrainOptions = trainingOptions('adam', ...
        MaxEpochs        = 200, ...
        MiniBatchSize    = 32, ...
        Shuffle          = 'every-epoch', ...
        InitialLearnRate = 5e-3, ...
        L2Regularization = 1e-4, ...
        Verbose          = false, ...
        Plots            = 'none');

    % train a small backbone with trainnet
    trainT1Hot = full(ind2vec(trainT'));   % 2 x N one-hot
    disp('Pre-training backbone with trainnet (gradient descent)')
    pretrained = trainnet(trainX, trainT1Hot', backboneAndHead, ...
                 'crossentropy', pretrainOptions);

    % strip the temporary classification head
    backbone = removeLayers(pretrained, {'fc_out', 'softmax'});

    if useScaling
        % fit element-wise linear scaling on backbone outputs
        backbone = initialize(backbone);
        rawFeatures = double(extractdata(predict(backbone, ...
            dlarray(trainX', 'CB'))));
        featureMin = min(rawFeatures, [], 2);
        featureMax = max(rawFeatures, [], 2);
        slope = 0.5 ./ (featureMax - featureMin);
        offset = 0.25 - slope .* featureMin;
        scaleToInner = @(x) min(max(slope .* x + offset, 0), 1);
        backbone = replaceLayer(backbone, 'scale_features', ...
            functionLayer(scaleToInner, Name = 'scale_features'));
    end

    % set additional parameters, if required
    netArgs = {};
    if strcmp(netType, 'Hypersphere_TopoART_C')
        netArgs = {'R', R};
    end

    % train the (Hypersphere) TopoART-C head
    ta = topoARTClassificationLayer(featureLen, moduleNum, rho_a, ...
        netType, netArgs{:}, Name = 'topoart_head');

    % build a datastore over (rows of trainX, rows of trainT) pairs
    dsX = arrayDatastore(trainX);
    dsT = arrayDatastore(trainT);
    ds  = combine(dsX, dsT);

    disp(['Training TopoART head on frozen backbone (incremental, not '...
          'gradient-based)'])
    net = trainTopoART(backbone, ta, ds, MaxEpochs = 1, ...
                       MiniBatchSize = 32);

    % predict on a uniform grid and plot
    gridMin = -1.5;
    gridMax = 2.5;
    [xg, yg] = meshgrid(gridMin:0.04:gridMax, gridMin:0.04:gridMax);
    gridData = [xg(:) yg(:)];

    if showSoftmaxPrediction

        % predict using the original head (fc_out + softmax)
        smY = predict(pretrained, dlarray(gridData', 'CB'));
        smY = double(extractdata(smY));
        [confSoftmax, classIDsSoftmax] = max(smY, [], 1);
        classIDsSoftmax(confSoftmax < threshSM) = 0;

        plotResults('Classification Results (Original Softmax Head)', ...
            gridData, classIDsSoftmax, samples)

    end

    % predict using the new TopoART head
    taY = predict(net, dlarray(gridData', 'CB'));
    taY = double(extractdata(taY));
    classIDsTopoArt = taY(1, :);
    confidencesTopoArt = taY(2, :);
    classIDsTopoArt(confidencesTopoArt < threshTA) = 0;

    plotResults('Classification Results (TopoART Head)', ...
        gridData, classIDsTopoArt, samples)

end

function plotResults(figName, gridData, classIDs, samples)

    figure(Name = figName)
    hold on
    grid
    axis([min(gridData(:, 1)) max(gridData(:, 1)) ...
          min(gridData(:, 2)) max(gridData(:, 2))])
    set(gca, 'fontsize', 15)
    title('classified grid points')

    colors = [0 0 1; 1 0 0];
    for i = 1:length(gridData)
        if classIDs(i) > 0
            plot(gridData(i, 1), gridData(i, 2), 's', ...
                'MarkerFaceColor', colors(classIDs(i), :), ...
                'MarkerEdgeColor', [1 1 1])
        end
    end

    for i = 1:length(samples)
        if samples(i, 3) == 1
            plot(samples(i, 1), samples(i, 2), '*k')
        else
            plot(samples(i, 1), samples(i, 2), 'ok')
        end
    end

end