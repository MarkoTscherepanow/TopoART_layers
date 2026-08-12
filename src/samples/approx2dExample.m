%APPROX2DEXAMPLE - Regression sample using topoARTRegressionLayer
%   ATTENTION: This function requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs (in the parent folder)
%   must be run before APPROX2DEXAMPLE can be used.
%
%   APPROX2DEXAMPLE demonstrates the use of the custom deep learning layer
%   topoARTRegressionLayer. A minimal dlnetwork is built that consists of
%   a featureInputLayer followed by a topoARTRegressionLayer. As TopoART
%   neural networks are not trained via gradient descent, the layer is
%   trained explicitly through its method learn (incremental learning)
%   before the dlnetwork is assembled. The network learns the MATLAB
%   peaks function z = peaks(x, y) from randomly drawn samples and then
%   approximates it on a uniform grid. The true surface and the
%   approximated surface are plotted in separate figures.
%
%   Both the inputs and the targets are rescaled to [0, 1], as required
%   by TopoART-R; the affine rescaling of the targets is inverted to
%   restore the original scale of the predictions.
%
%   Syntax
%     APPROX2DEXAMPLE
%     APPROX2DEXAMPLE(rho_a)
%
%   Input Arguments
%     rho_a - Vigilance parameter of the first module
%       It must lie in the interval [0, 1] and controls the category size
%       and thereby the resolution of the approximation: higher values
%       yield smaller categories and a finer surface but need more
%       training samples. (default: 0.95)
function approx2dExample(rho_a)

    narginchk(0, 1)
    nargoutchk(0, 0)

    if nargin < 1
        rho_a = 0.95;
    end

    if rho_a < 0 || rho_a > 1
        error('rho_a must have a value from the interval [0, 1].')
    end

    % length of the input vector (x and y coordinates)
    inputLen = 2;

    % length of the output vector (function value z)
    outputLen = 1;

    % number of TopoART modules (default: 2)
    moduleNum = 2;

    % choose the wrapped TopoART-R network (The string is resolved to a
    % LibTopoART.Compatibility.Network enum inside the constructor, after
    % the .NET assembly has been loaded on demand.)
    netType = 'Fast_TopoART_R';

    % number of randomly drawn training samples
    trainNum = 3000;

    % The peaks function is evaluated on [-3, 3] x [-3, 3], where its
    % values lie within (-7, 8.5). All bounds are decided in advance, so
    % the rescaling to [0, 1] (required by TopoART-R for inputs and
    % outputs alike) is a deterministic transform whose inverse restores
    % the original scale of the predictions.
    xyMin = -3;
    xyMax =  3;
    zMin  = -7;
    zMax  =  8.5;

    surfColormap = 'hot';

    % disable/enable export of the result figures as PNG images into the
    % images folder in the repository root
    exportImages = false;

    % Set path so that the layer and the helpers can be located when
    % the example is run from the samples folder. The wrapped .NET
    % library is loaded on demand by the constructor of
    % topoARTRegressionLayer. onCleanup restores the user's path on
    % normal exit, error, or Ctrl+C.
    basePath = fileparts(mfilename('fullpath'));
    srcPath  = fileparts(basePath);
    oldPath  = addpath(srcPath, fullfile(srcPath, 'helpers'));
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % map image names to files in the images/regressor folder; an empty
    % file name disables the PNG export in plotSurface
    if exportImages
        imagesPath = fullfile(fileparts(srcPath), 'images', ...
            'regressor'); %#ok<UNRCH>
        if ~isfolder(imagesPath)
            mkdir(imagesPath)
        end
        imageFile = @(name) fullfile(imagesPath, name);
    else
        imageFile = @(name) ''; %#ok<UNRCH>
    end

    % draw random training points and evaluate the peaks function
    trainXY = xyMin + (xyMax - xyMin) * rand(trainNum, 2);
    trainZ  = peaks(trainXY(:, 1), trainXY(:, 2));

    % rescale the inputs and targets to [0, 1]
    trainX = (trainXY - xyMin) / (xyMax - xyMin);
    trainT = (trainZ - zMin) / (zMax - zMin);

    % construct the custom layer (instantiates the wrapped .NET network)
    tarLayer = topoARTRegressionLayer(inputLen, outputLen, ...
        moduleNum, rho_a, netType);

    disp('Start training (incremental, not gradient-based)')
    tic
    tarLayer.learn(trainX, trainT);
    toc

    % assemble a minimal dlnetwork: input layer + TopoART-R head
    layers = [
        featureInputLayer(inputLen, Name = 'input')
        tarLayer
    ];
    net = dlnetwork(layers);

    % Generate a uniform grid of test points and its rescaled version.
    [xg, yg] = meshgrid(xyMin:0.25:xyMax);
    zTrue = peaks(xg, yg);
    gridData = ([xg(:) yg(:)] - xyMin) / (xyMax - xyMin);

    disp('Start prediction')
    tic
    Y = predict(net, dlarray(gridData', 'CB'));
    toc

    % restore the original scale of the predictions by inverting the
    % target rescaling; a NaN signals that no prediction was possible
    zPred = double(extractdata(Y)) * (zMax - zMin) + zMin;
    zPred = reshape(zPred, size(xg));

    % report the approximation quality on the grid
    fprintf('Mean absolute error on the grid: %.3f\n', ...
        mean(abs(zPred(:) - zTrue(:)), 'omitnan'))
    nanNum = nnz(isnan(zPred));
    if nanNum > 0
        fprintf('No prediction was possible for %d grid points.\n', ...
            nanNum)
    end

    disp('Compute results figures')

    plotSurface('True Function (peaks)', 'peaks function', xg, yg, ...
        zTrue, [zMin zMax], surfColormap, imageFile('peaks.png'))

    plotSurface('Approximation Results (topoARTRegressionLayer)', ...
        'TopoART-R approximation', xg, yg, zPred, [zMin zMax], ...
        surfColormap, imageFile('TopoART_approximation.png'))

end
