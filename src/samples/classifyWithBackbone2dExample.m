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
%     5. After this inference stage, the TopoART-C head is trained
%        incrementally with a third class that was not part of the initial
%        training, and the grid prediction is repeated. The backbone and
%        its scaling transform remain frozen; only the TopoART-C head
%        extends its knowledge.
%
%   The two-dimensional dataset used here contains two classes distributed
%   as two interleaved half-moons. The third class learnt incrementally
%   has a bimodal distribution whose modes lie in regions the half-moons
%   leave free (upper right and lower left).
%
%   Syntax
%     CLASSIFYWITHBACKBONE2DEXAMPLE
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling)
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling, threshTA)
%     CLASSIFYWITHBACKBONE2DEXAMPLE(useScaling, threshTA, threshSM)
%
%   Input Arguments
%     useScaling - Select input normalization method (scaling or tanh-based)
%       TopoART inputs must lie in [0, 1]. This switch allows choosing
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
    R = sqrt(featureLen) / 2; %#ok<NASGU>

    % choose the wrapped (Hypersphere) TopoART-C network (The string is
    % resolved to a LibTopoART.Compatibility.Network enum inside the
    % constructor, after the .NET assembly has been loaded on demand.)
    netType = 'Fast_TopoART_C';

    % disable/enable prediction using the original head (fc_out + softmax)
    showSoftmaxPrediction = true;

    % disable/enable export of the result figures as PNG images into the
    % images folder in the repository root (used by the README)
    exportImages = false;

    % Set path so that the layer and helpers can be located when the
    % example is run from the samples folder (The wrapped .NET library is
    % loaded on demand by the constructor of topoARTClassificationLayer.)
    % onCleanup restores the user's path on normal exit, error, or Ctrl+C.
    basePath = fileparts(mfilename('fullpath'));
    srcPath  = fileparts(basePath);
    oldPath  = addpath(srcPath, fullfile(srcPath, 'helpers'));
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % map image names to files in the images/classifier folder; an empty
    % file name disables the PNG export in plotResults
    if exportImages
        imagesPath = fullfile(fileparts(srcPath), 'images', ...
            'classifier'); %#ok<UNRCH>
        if ~isfolder(imagesPath)
            mkdir(imagesPath)
        end
        imageFile = @(name) fullfile(imagesPath, name);
    else
        imageFile = @(name) ''; %#ok<UNRCH>
    end

    % seed the random number generator so the moon noise, the sample
    % shuffle, and the backbone weight initialization are reproducible
    % (drop or change the seed to explore initialization variability)
    rng(0)

    % generate and prepare the training data
    samples = create2dMoons;
    samples = samples(randperm(length(samples)), :);

    trainX = samples(:, 1:2);
    trainT = samples(:, 3); % class IDs in {1, 2}

    if useScaling
        scaleToInner = @(x) x; % replaced after pretraining
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
    trainT1Hot = full(ind2vec(trainT')); % 2 x N one-hot
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
        netArgs = {'R', R}; %#ok<UNRCH>
    end

    % construct the (Hypersphere) TopoART-C head
    ta = topoARTClassificationLayer(featureLen, moduleNum, rho_a, ...
        netType, netArgs{:}, Name = 'topoart_head');

    % build a datastore over (rows of trainX, rows of trainT) pairs
    dsX = arrayDatastore(trainX);
    dsT = arrayDatastore(trainT);
    ds  = combine(dsX, dsT);

    disp(['Training TopoART head on frozen backbone (incremental, not ' ...
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
        smY = predict(pretrained, dlarray(gridData', 'CB')); %#ok<UNRCH>
        smY = double(extractdata(smY));
        [confSoftmax, classIDsSoftmax] = max(smY, [], 1);
        classIDsSoftmax(confSoftmax < threshSM) = 0;

        plotResults('Classification Results (Original Softmax Head)', ...
            gridData, classIDsSoftmax, samples, ...
                imageFile('softmax_head.png'))

    end

    % predict using the new TopoART head
    taY = predict(net, dlarray(gridData', 'CB'));
    taY = double(extractdata(taY));
    classIDsTopoArt = taY(1, :);
    confidencesTopoArt = taY(2, :);
    classIDsTopoArt(confidencesTopoArt < threshTA) = 0;

    plotResults('Classification Results (TopoART Head)', ...
        gridData, classIDsTopoArt, samples, imageFile('TopoART_head.png'))

    % demonstrate incremental learning (After the inference stage above,
    % a third class with two well-separated modes is added in regions
    % the moons leave free; the backbone and the scaling transform stay
    % frozen so that the feature space of the existing prototypes is
    % preserved.)
    newX = [create2dGaussian([1.25 1.0]); create2dGaussian([-0.25 -0.5])];
    newT = 3 * ones(size(newX, 1), 1);

    dsNew = combine(arrayDatastore(newX), arrayDatastore(newT));

    disp('Extending TopoART head with a new class (incremental training)')
    net = trainTopoART(backbone, ta, dsNew, MaxEpochs = 1, ...
                       MiniBatchSize = 32);

    % repeat the grid prediction with the extended TopoART head
    taY = predict(net, dlarray(gridData', 'CB'));
    taY = double(extractdata(taY));
    classIDsTopoArt = taY(1, :);
    confidencesTopoArt = taY(2, :);
    classIDsTopoArt(confidencesTopoArt < threshTA) = 0;

    plotResults(['Classification Results (TopoART Head After ' ...
        'Incremental Training)'], gridData, classIDsTopoArt, ...
        [samples; newX newT], ...
        imageFile('TopoART_head_incremental.png'))

end

function plotResults(figName, gridData, classIDs, samples, exportFile)
%PLOTRESULTS - Plot classified grid points and training samples
%   Opens a figure named figName and draws the grid points as squares
%   colored according to their entries in classIDs; points with a class ID
%   of 0 (rejected by the confidence threshold) are omitted. The training
%   samples are overlaid using class-specific black markers. If exportFile
%   is given and non-empty, the figure size is fixed and the finished
%   figure is exported to exportFile as a PNG image.

    narginchk(4, 5)
    nargoutchk(0, 0)

    if nargin < 5
        exportFile = '';
    end

    if isempty(exportFile)
        figure(Name = figName)
    else
        % fix the figure size so that exported images are identical
        % across screens
        figure(Name = figName, Position = [100 100 700 560])
    end

    hold on
    grid
    axis([min(gridData(:, 1)) max(gridData(:, 1)) ...
          min(gridData(:, 2)) max(gridData(:, 2))])
    set(gca, 'fontsize', 15)
    title('classified grid points')

    colors = [0 0 1; 1 0 0; 0 0.8 0];
    for i = 1:length(gridData)
        if classIDs(i) > 0
            plot(gridData(i, 1), gridData(i, 2), 's', ...
                'MarkerFaceColor', colors(classIDs(i), :), ...
                'MarkerEdgeColor', [1 1 1])
        end
    end

    markers = {'*k', 'ok', 'pk'};
    for i = 1:length(samples)
        plot(samples(i, 1), samples(i, 2), markers{samples(i, 3)})
    end

    if ~isempty(exportFile)
        exportgraphics(gcf, exportFile, Resolution = 100)
    end

end

function pts = create2dGaussian(center, sampleNum, noise)
%CREATE2DGAUSSIAN - Generate a single isotropic Gaussian mode
%   Returns sampleNum data points (format: x, y) drawn around center
%   (format: x, y) with standard deviation noise in each direction.

    narginchk(1, 3)
    nargoutchk(1, 1)

    if nargin < 2
        sampleNum = 75;
    end

    if nargin < 3
        noise = 0.1;
    end

    pts = center + noise * randn(sampleNum, 2);

end