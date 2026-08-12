%APPROXWITHBACKBONE2DEXAMPLE - Train TopoART-R on top of a frozen backbone
%   ATTENTION: This function requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs (in the parent folder)
%   must be run before APPROXWITHBACKBONE2DEXAMPLE can be used.
%
%   This example demonstrates the typical end-to-end regression workflow
%   with a backbone:
%     1. A small backbone is trained with trainnet on samples of the
%        MATLAB peaks function using a temporary differentiable head (a
%        fully connected output layer trained with an MSE loss). This
%        stage uses standard gradient descent. A circular region around
%        the highest peak is omitted from the training data.
%     2. The differentiable head is stripped off, leaving the trained
%        feature extractor.
%     3. The TopoART-R head is constructed and trained on the frozen
%        backbone via trainTopoART, using the same samples. This stage
%        uses incremental TopoART learning, not gradient descent.
%     4. The true surface and the approximations of both heads are
%        plotted; the overlaid training samples make the omitted
%        region visible. Neither head has ever seen the omitted
%        region, so both can only interpolate across it and miss the
%        peak.
%     5. The TopoART-R head is trained incrementally on samples from
%        the omitted region, and its plot is repeated with the new
%        samples included in the overlay. The backbone and
%        its scaling transform remain frozen; only the TopoART-R head
%        extends its knowledge, without forgetting the rest of the
%        domain and without any gradient retraining. The pretrained
%        head cannot follow without such retraining. Because the
%        omitted region is enclosed by training data, the frozen
%        backbone interpolates its features there instead of
%        extrapolating: they stay distinct from the rest of the
%        domain, which protects the existing knowledge, but they vary
%        less than in trained regions, which limits the accuracy of
%        the recovered peak. A wide feature vector and the smooth tanh
%        activations counteract this limit.
%
%   The backbone features are mapped into [0, 1] by a frozen transform
%   selected via useScaling, as required by TopoART-R. The regression
%   targets are rescaled to [0, 1] with bounds decided in advance that
%   cover the whole domain; the rescaling is inverted to restore the
%   original scale of the predictions.
%
%   Syntax
%     APPROXWITHBACKBONE2DEXAMPLE
%     APPROXWITHBACKBONE2DEXAMPLE(useScaling)
%
%   Input Arguments
%     useScaling - Select input normalisation method (scaling or tanh-based)
%       TopoART inputs must lie in [0, 1]. This switch allows choosing
%       between linear scaling (if true) and tanh-based normalisation.
%       (default: true)
function approxWithBackbone2dExample(useScaling)

    narginchk(0, 1)
    nargoutchk(0, 0)

    if nargin < 1
        useScaling = true;
    end

    % raw input dimensionality (x and y coordinates)
    inputLen = 2;

    % Width of the backbone's tanh hidden layers. The smooth tanh
    % activations interpolate with nonvanishing gradients, so the
    % features keep varying inside the omitted region.
    hiddenLen = 32;

    % Backbone output dimensionality (TopoART-R input). A wide feature
    % vector helps the incremental stage: the more feature dimensions,
    % the more of them still vary inside the omitted region. It must
    % not exceed hiddenLen, though, or the features become redundant.
    featureLen = 32;

    % length of the output vector (function value z)
    outputLen = 1;

    % number of TopoART modules (default: 2)
    moduleNum = 2;

    % vigilance parameter of the first module
    rho_a = 0.97;

    % choose the wrapped TopoART-R network (The string is resolved to a
    % LibTopoART.Compatibility.Network enum inside the constructor, after
    % the .NET assembly has been loaded on demand.)
    netType = 'Fast_TopoART_R';

    % number of training samples drawn outside the omitted region
    trainNum = 2000;

    % number of training samples drawn from the omitted region for the
    % incremental stage
    newNum = 500;

    % centre and radius of the circular region around the highest peak
    % that is omitted from backbone pretraining and from the initial
    % TopoART-R training; it is learnt only incrementally
    holeCentre = [0 1.6];
    holeRadius = 1;

    % The peaks function is evaluated on [-3, 3] x [-3, 3], where its
    % values lie within (-7, 8.5). All bounds are decided in advance and
    % cover the whole domain, including the omitted region that is
    % learnt only incrementally, so the target rescaling to [0, 1]
    % (required by TopoART-R) stays frozen throughout; its inverse
    % restores the original scale of the predictions.
    xyMin = -3;
    xyMax =  3;
    zMin  = -7;
    zMax  =  8.5;

    surfColormap = 'hot';

    % disable/enable prediction using the temporary pretrained head
    % (fc_out)
    showPretrainedHeadPrediction = true;

    % disable/enable export of the result figures as PNG images into the
    % images folder in the repository root
    exportImages = false;

    % Set path so that the layer, trainTopoART, and the helpers can be
    % located when the example is run from the samples folder. The
    % wrapped .NET library is loaded on demand by the constructor of
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

    % seed the random number generator so the sample draw and the
    % backbone weight initialisation are reproducible (drop or change
    % the seed to explore initialisation variability)
    rng(0)

    % Draw random training points and evaluate the peaks function.
    % Backbone pretraining and the initial TopoART-R training use only
    % the points outside the omitted region; the region itself is
    % learnt incrementally later on. The targets are rescaled to
    % [0, 1]; the unscaled function values are kept for the sample
    % overlays in the result plots.
    trainX = drawRegionSamples(trainNum, xyMin, xyMax, holeCentre, ...
        holeRadius, false);
    trainZ = peaks(trainX(:, 1), trainX(:, 2));
    trainT = (trainZ - zMin) / (zMax - zMin);
    newX = drawRegionSamples(newNum, xyMin, xyMax, holeCentre, ...
        holeRadius, true);
    newZ = peaks(newX(:, 1), newX(:, 2));
    newT = (newZ - zMin) / (zMax - zMin);

    if useScaling
        scaleToInner = @(x) x; % replaced after pretraining
    else
        scaleToInner = @(x) 0.5 + 0.5 * tanh(x);
    end

    % set topology of the base network (backbone + head)
    backboneAndHead = [
        featureInputLayer(inputLen,     Name = 'input')
        fullyConnectedLayer(hiddenLen,  Name = 'fc1')
        tanhLayer(                      Name = 'tanh1')
        fullyConnectedLayer(hiddenLen,  Name = 'fc2')
        tanhLayer(                      Name = 'tanh2')
        fullyConnectedLayer(featureLen, Name = 'features')
        % satisfy TopoART input range precondition
        functionLayer(scaleToInner,     Name = 'scale_features')
        fullyConnectedLayer(outputLen,  Name = 'fc_out')
    ];

    pretrainOptions = trainingOptions('adam', ...
        MaxEpochs        = 200, ...
        MiniBatchSize    = 32, ...
        Shuffle          = 'every-epoch', ...
        InitialLearnRate = 5e-3, ...
        L2Regularization = 1e-4, ...
        Verbose          = false, ...
        OutputFcn        = @trainnetDots, ...
        Plots            = 'none');

    % train a small backbone with trainnet
    disp('Pre-training backbone with trainnet (gradient descent)')
    pretrained = trainnet(trainX, trainT, backboneAndHead, 'mse', ...
                          pretrainOptions);

    % strip the temporary regression head
    backbone = removeLayers(pretrained, 'fc_out');

    if useScaling
        % fit element-wise linear scaling on backbone outputs
        backbone = initialize(backbone);
        rawFeatures = double(extractdata(predict(backbone, ...
            dlarray(trainX', 'CB'))));
        featureMin = min(rawFeatures, [], 2);
        featureMax = max(rawFeatures, [], 2);
        span = featureMax - featureMin;
        span(span < 1e-6) = 1;
        slope = 0.5 ./ span;
        offset = 0.25 - slope .* featureMin;
        scaleToInner = @(x) min(max(slope .* x + offset, 0), 1);
        backbone = replaceLayer(backbone, 'scale_features', ...
            functionLayer(scaleToInner, Name = 'scale_features'));
    end

    % construct the TopoART-R head
    tarLayer = topoARTRegressionLayer(featureLen, outputLen, ...
        moduleNum, rho_a, netType, Name = 'topoart_head');

    % build a datastore over (rows of trainX, rows of trainT) pairs
    ds = combine(arrayDatastore(trainX), arrayDatastore(trainT));

    disp(['Training TopoART-R head on frozen backbone (incremental, ' ...
          'not gradient-based)'])
    net = trainTopoART(backbone, tarLayer, ds, MaxEpochs = 1, ...
                       MiniBatchSize = 32);

    % generate a uniform grid of test points covering the whole domain
    [xg, yg] = meshgrid(xyMin:0.25:xyMax);
    zTrue = peaks(xg, yg);
    gridData = [xg(:) yg(:)];
    insideIdx = hypot(xg - holeCentre(1), yg - holeCentre(2)) ...
        <= holeRadius;

    % The overlaid training samples make the omitted region visible in
    % every result plot.
    plotSurface('True Function (peaks)', 'peaks function', xg, yg, ...
        zTrue, [zMin zMax], surfColormap, ...
        imageFile('peaks_training_distribution.png'), [trainX trainZ])

    if showPretrainedHeadPrediction

        % Predict using the temporary pretrained head. Its predictions
        % do not change any more; learning the omitted region would
        % require gradient retraining.
        headY = predict(pretrained, ...
            dlarray(gridData', 'CB')); %#ok<UNRCH>
        zPredHead = restoreScale(headY, zMin, zMax, size(xg));
        reportErrors('pretrained head', zTrue, zPredHead, insideIdx)

        plotSurface('Approximation Results (Pretrained Head)', ...
            'pretrained head', xg, yg, zPredHead, [zMin zMax], ...
            surfColormap, imageFile('pretrained_head.png'), ...
            [trainX trainZ])

    end

    % predict using the new TopoART-R head
    tarY = predict(net, dlarray(gridData', 'CB'));
    zPredTar = restoreScale(tarY, zMin, zMax, size(xg));
    reportErrors('TopoART-R head ', zTrue, zPredTar, insideIdx)

    plotSurface('Approximation Results (TopoART-R Head)', ...
        'TopoART-R head', xg, yg, zPredTar, [zMin zMax], ...
        surfColormap, imageFile('TopoART_head.png'), [trainX trainZ])

    % demonstrate incremental learning (After the inference stage above,
    % the omitted region is added; the backbone and the scaling
    % transform stay frozen so that the feature space of the existing
    % prototypes is preserved.)
    dsNew = combine(arrayDatastore(newX), arrayDatastore(newT));

    disp('Extending TopoART-R head to the omitted region (incremental)')
    net = trainTopoART(backbone, tarLayer, dsNew, MaxEpochs = 1, ...
                       MiniBatchSize = 32);

    % repeat the grid prediction with the extended TopoART-R head
    tarY = predict(net, dlarray(gridData', 'CB'));
    zPredTar = restoreScale(tarY, zMin, zMax, size(xg));
    reportErrors('TopoART-R head ', zTrue, zPredTar, insideIdx)

    plotSurface(['Approximation Results (TopoART-R Head After ' ...
        'Incremental Training)'], 'After incremental training', xg, yg, ...
        zPredTar, [zMin zMax], surfColormap, ...
        imageFile('TopoART_head_incremental.png'), ...
        [trainX trainZ; newX newZ])

end

function xy = drawRegionSamples(sampleNum, xyMin, xyMax, ...
        holeCentre, holeRadius, inside)
%DRAWREGIONSAMPLES - Draw uniform points wrt. a circular region
%   Returns sampleNum data points (format: x, y) drawn uniformly from
%   the square domain [xyMin, xyMax] x [xyMin, xyMax], keeping only
%   the points inside the circle around holeCentre with radius
%   holeRadius (inside true) or outside of it (inside false).

    xy = zeros(0, 2);
    while size(xy, 1) < sampleNum
        cand = xyMin + (xyMax - xyMin) * rand(sampleNum, 2);
        inHole = hypot(cand(:, 1) - holeCentre(1), ...
            cand(:, 2) - holeCentre(2)) <= holeRadius;
        if inside
            cand = cand(inHole, :);
        else
            cand = cand(~inHole, :);
        end
        xy = [xy; cand]; %#ok<AGROW>
    end
    xy = xy(1:sampleNum, :);

end

function z = restoreScale(Y, zMin, zMax, gridSize)
%RESTORESCALE - Restore the original scale of scaled predictions
%   Extracts the predictions from the dlarray Y, inverts the target
%   rescaling to [0, 1], and reshapes the result to gridSize. A NaN
%   signals that no prediction was possible.

    z = double(extractdata(Y)) * (zMax - zMin) + zMin;
    z = reshape(z, gridSize);

end

function reportErrors(name, zTrue, zPred, insideIdx)
%REPORTERRORS - Report the mean absolute error per domain region
%   Prints the mean absolute error of the approximation zPred wrt. the
%   true surface zTrue, separately for the omitted region (insideIdx
%   true) and the rest of the domain.

    fprintf('%s: MAE %.3f (omitted region), %.3f (rest)\n', name, ...
        mean(abs(zPred(insideIdx) - zTrue(insideIdx)), 'omitnan'), ...
        mean(abs(zPred(~insideIdx) - zTrue(~insideIdx)), 'omitnan'))

end

