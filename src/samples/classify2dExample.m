%CLASSIFY2DEXAMPLE - Classification sample using topoARTClassificationLayer
%   ATTENTION: This function can only be executed on Windows or macOS
%   systems and requires .NET Framework 4.7.2 or higher, or .NET 6.0 or
%   higher. Furthermore, installLibs (in the parent src folder) must be run
%   before CLASSIFY2DEXAMPLE can be used.
%
%   CLASSIFY2DEXAMPLE demonstrates the use of the custom deep learning
%   layer topoARTClassificationLayer. A minimal dlnetwork is built that
%   consists of a featureInputLayer followed by a
%   topoARTClassificationLayer. As TopoART neural networks are not trained
%   via gradient descent, the layer is trained explicitly through its
%   method learn (incremental learning) before the dlnetwork is assembled.
%   The trained network is then used to predict class IDs on a uniform grid
%   of test points. The results are plotted.
%
%   Two datasets are supported: two intertwined spirals and two
%   interleaved half-moons.
%
%   Syntax
%     CLASSIFY2DEXAMPLE
%     CLASSIFY2DEXAMPLE(dataset)
%     CLASSIFY2DEXAMPLE(dataset, confThresh)
%
%   Input Arguments
%     dataset - Dataset to be used
%       Allowed values are 'spirals' (two intertwined spirals via
%       create2dSpirals) and 'moons' (two interleaved half-moons via
%        create2dMoons). (default: 'spirals')
%     confThresh - Confidence threshold
%       The classification confidence lies in the range [0, 1] where a
%       confidence of 1 signifies that an input is completely known by the
%       network. As a consequence, the region of the input space assigned
%       to a class increases if confThresh is reduced. On the other hand,
%       the classification results become more uncertain. (default: 0.96)
function classify2dExample(dataset, confThresh)

    narginchk(0, 2)
    nargoutchk(0, 0)

    if nargin < 1
        dataset = 'spirals';
    end

    dataset = validatestring(dataset, {'spirals', 'moons'});

    if nargin < 2
        confThresh = 0.96;
    end
   
    if confThresh < 0 || confThresh > 1
        error('confThresh must have a value from the interval [0, 1].')
    end

    % length of the input vectors
    inputLen = 2;

    % number of TopoART modules (default: 2)
    moduleNum = 2;

    % vigilance parameter of the first module
    rho_a = 0.95;

    % radial extend R required by Hypersphere TopoART-C, set according to
    % R = sqrt(inputDimension * (dataMax - dataMin)^2) / 2
    % where (dataMax - dataMin) is the per-dimension range of the actual
    % TopoART input (Inputs are rescaled to [0, 1] below, so the
    % post-rescale bounds 0 and 1 are used here (not the raw dataMin /
    % dataMax of the chosen dataset).)
    R = sqrt(inputLen * (1 - 0)^2) / 2; %#ok<NASGU>

    % choose the wrapped (Hypersphere) TopoART-C network (The string is
    % resolved to a LibTopoART.Compatibility.Network enum inside the
    % constructor, after the .NET assembly has been loaded on demand.)
    netType = 'Fast_TopoART_C';

    % Set path so that the layer and the helpers can be located when the
    % example is run from the samples folder. The wrapped .NET library is
    % loaded on demand by the constructor of topoARTClassificationLayer.
    % onCleanup restores the user's path on normal exit, error, or Ctrl+C.
    basePath = fileparts(mfilename('fullpath'));
    srcPath  = fileparts(basePath);
    oldPath  = addpath(srcPath, fullfile(srcPath, 'helpers'));
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % Generate the chosen dataset and shuffle it randomly. The min/max
    % bounds are dataset-specific but decided in advance, so the rescaling
    % to [0, 1] (required by TopoART) is a deterministic transform.
    switch dataset
        case 'spirals'
            samples = create2dSpirals;
            dataMin = -7.0;
            dataMax =  7.0;
        case 'moons'
            samples = create2dMoons;
            dataMin = -1.5;
            dataMax =  2.5;
    end

    samples = samples(randperm(length(samples)), :);
    samples(:, 1:2) = (samples(:, 1:2) - dataMin) / (dataMax - dataMin);

    trainX = samples(:, 1:2);
    trainT = samples(:, 3);   % class IDs in {1, 2}

    % set additional parameters, if required
    netArgs = {};
    if strcmp(netType, 'Hypersphere_TopoART_C')
        netArgs = {'R', R}; %#ok<UNRCH>
    end

    % construct the custom layer (instantiates the wrapped .NET network)
    topoArt = topoARTClassificationLayer(inputLen, moduleNum, rho_a, ...
                                         netType, netArgs{:});

    disp('Start training (online, not gradient-based)')
    tic
    topoArt.learn(trainX, trainT);
    toc

    % assemble a minimal dlnetwork: input layer + TopoART head
    layers = [
        featureInputLayer(inputLen, Name = 'input')
        topoArt
    ];
    net = dlnetwork(layers);

    % generate a uniform grid of test points
    [xg, yg] = meshgrid(0:0.01:1, 0:0.01:1);
    gridData = [xg(:) yg(:)];

    disp('Start prediction')
    tic
    Y = predict(net, dlarray(gridData', 'CB'));
    toc

    Y = double(extractdata(Y));      % 2-by-N: [classID; confidence]
    classIDs = Y(1, :);
    confidences = Y(2, :);

    % apply confidence threshold and mark rejected points as unknown (0)
    classIDs(confidences < confThresh) = 0;

    disp('Compute results figure')

    figure(Name = 'Classification Results (topoARTClassificationLayer)')
    hold on
    grid
    axis([0 1 0 1])
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

    % overlay the training samples
    for i = 1:length(trainX)
        if trainT(i) == 1
            plot(trainX(i, 1), trainX(i, 2), '*k')
        else
            plot(trainX(i, 1), trainX(i, 2), 'ok')
        end
    end

end