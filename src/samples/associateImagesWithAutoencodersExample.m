%ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE - Associate images via autoencoders
%   ATTENTION: This function requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs (in the parent folder) and
%   getDatasets (in this folder) must be run before
%   ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE can be used.
%
%   ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE demonstrates the custom deep
%   learning layer topoARTAssociativeMemoryLayer as a bidirectional
%   associative memory on top of two autoencoders. The dataset contains two
%   kinds of images grouped into owners and objects, with an m-to-n mapping
%   between them: each owner can be associated with several objects and an
%   object can be shared by several owners.
%
%   In contrast to associateImagesExample, the keys are not the raw images
%   but compact latent codes:
%     1. One convolutional autoencoder is trained per key, each on
%        its own native image size (owner and object images differ in
%        size and are not resized; the decoder only trims a few surplus
%        pixels for alignment). The bottleneck is linear and a frozen
%        affine rescale maps each latent into [0.25, 0.75], a sub-interval
%        of the [0, 1] range TopoART requires. The encoder applies this
%        rescale, so its output is the TopoART key; the decoder inverts it
%        first, so a recalled key decodes without any extra transform.
%     2. The TopoART-AM head learns associations between owner latents
%        (key1) and object latents (key2) via trainTopoARTAM, which uses
%        the two frozen encoders as backbones. This stage uses incremental
%        TopoART learning, not gradient descent.
%     3. Recall is demonstrated in both directions. A test owner is
%        presented to recall its objects, and a test object to recall its
%        owners, by switching Direction on the shared head. Recall is
%        a 1-to-n mapping, so recall returns the full set of associated
%        latents; each is turned back into an image by the matching
%        decoder. No training images are needed at inference.
%
%   Robustness: TopoART-AM may be sensitive to its input, so small pixel
%   changes can alter which associations a recall returns (see
%   associateImagesExample). Compressing each image into a few latent
%   variables removes most of that sensitivity: a small input perturbation
%   barely moves the compact latent key, while it shifts a raw-pixel key
%   directly. The associations are therefore far more stable here than in
%   the raw-pixel sample.
%
%   The dataset is the one from the LibTopoART TopoART-AM image-association
%   sample (TopoART-AM_sample2), with its images converted from JPG to PNG;
%   the owner-to-object mapping is taken from there as well.
%
%   Details on bidirectional associations with TopoART-AM can be found in:
%
%   "Marko Tscherepanow, Marco Kortkamp, and Marc Kammer (2011). A
%   Hierarchical ART Network for the Stable Incremental Learning of
%   Topological Structures and Associations from Noisy Data. Neural
%   Networks 24(8): 906-916. Elsevier."
%
%   Syntax
%     ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE
%     ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE(recallActThresh)
%     ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE(recallActThresh, maxRecalls)
%     ASSOCIATEIMAGESWITHAUTOENCODERSEXAMPLE(recallActThresh, ...
%         maxRecalls, cacheFile)
%
%   Input Arguments
%     recallActThresh - Minimum F3 activation kept during recall
%       Recall is a 1-to-n mapping. A value of 0 returns every recalled
%       association; raising it keeps only strong matches. (range: [0, 1];
%       default: 0)
%     maxRecalls - Maximum number of recalled images shown per stimulus
%       (positive integer; default: 5)
%     cacheFile - MAT-file caching the trained autoencoders
%       The autoencoders do not depend on any TopoART-AM parameter, so the
%       slow gradient descent can be reused while the head is re-tuned. If
%       cacheFile exists and matches the current latentDim, image sizes,
%       and autoencoder training epochs, the autoencoders are loaded from
%       it; otherwise they are trained and saved to it. An empty value
%       disables caching. (character vector; default: '')
function associateImagesWithAutoencodersExample(recallActThresh, ...
        maxRecalls, cacheFile)

    narginchk(0, 3)
    nargoutchk(0, 0)

    if nargin < 1
        recallActThresh = 0;
    end

    if recallActThresh < 0 || recallActThresh > 1
        error(['recallActThresh must have a value from the ' ...
            'interval [0, 1].'])
    end

    if nargin < 2
        maxRecalls = 5;
    end

    if maxRecalls < 1 || maxRecalls ~= floor(maxRecalls)
        error('maxRecalls must be a positive integer.')
    end

    if nargin < 3
        cacheFile = '';
    end

    % number of TopoART modules
    moduleNum = 1;

    % vigilance parameter of the first module (It is coupled to the latent
    % length and must be tuned with it.)
    rho_a = 0.85;

    % learning rate of the second best-matching neuron (The high value
    % mainly benefits recall in the direction key2 -> key1.)
    betaSbm = 1.0;

    % phi and tau jointly control TopoART's noise-reduction mechanism: a
    % candidate neuron must be activated at least phi times within tau
    % learning steps to become permanent, so rarely hit (noisy) candidates
    % are purged.
    phi = 2;
    tau = 200;

    % autoencoder bottleneck = TopoART-AM key length. A longer latent
    % improves recall when rho_a is tuned to match.
    latentDim = 64;

    % training epochs for the autoencoders (gradient descent) and the
    % maximum for the incremental TopoART-AM training (it stops earlier
    % once the head has stabilised)
    aeMaxEpochs = 400;
    tamMaxEpochs = 25;

    % number of owners/objects and training images per item
    ownerNum = 6;
    objectNum = 20;
    imagesPerItem = 20;

    % m-to-n mapping of owners to objects (objects may be shared)
    trainMap = { ...
        [1 2 3 9 17 19], ... % owner 1
        [3 4 9 15 17], ... % owner 2
        [1 3 7 10 11 12 16 17], ... % owner 3
        [3 5 17 18 20], ... % owner 4
        [3 8 14 19 20], ... % owner 5
        [3 5 10 13 16]}; % owner 6

    % disable/enable export of the recall figures as PNG images into the
    % images folder in the repository root
    exportImages = false;

    % Locate the source folder and the dataset relative to this file so
    % the example can be run from the samples folder. The wrapped .NET
    % library is loaded on demand by the layer constructor. onCleanup
    % restores the user's path on normal exit, error, or Ctrl+C.
    basePath = fileparts(mfilename('fullpath'));
    srcPath  = fileparts(basePath);
    oldPath  = addpath(srcPath, fullfile(srcPath, 'helpers'));
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % map image names to files in the images/associative_memory folder;
    % an empty file name disables the PNG export in recallAndShow
    if exportImages
        imagesPath = fullfile(fileparts(srcPath), 'images', ...
            'associative_memory'); %#ok<UNRCH>
        if ~isfolder(imagesPath)
            mkdir(imagesPath)
        end
        imageFile = @(name) fullfile(imagesPath, name);
    else
        imageFile = @(name) ''; %#ok<UNRCH>
    end

    datasetPath = fullfile(basePath, 'data', 'ObjectsOwners_dataset');
    if ~isfolder(datasetPath)
        error('Dataset folder not found: %s', datasetPath)
    end

    ownerPrefix  = fullfile(datasetPath, 'train', 'owners', 'owner_');
    objectPrefix = fullfile(datasetPath, 'train', 'objects', 'object_');

    disp('Load training images')
    trainOwners  = loadImages(ownerPrefix, ownerNum, ...
        imagesPerItem, 'png');
    trainObjects = loadImages(objectPrefix, objectNum, ...
        imagesPerItem, 'png');

    % native image size per key (owners and objects differ in size);
    % the autoencoders keep these sizes and are not forced to a common one
    ownerSize  = size(trainOwners{1, 1});
    objectSize = size(trainObjects{1, 1});

    % seed the random number generator so the autoencoder weight
    % initialisation and epoch shuffling are reproducible (drop or
    % change the seed to explore initialisation variability)
    rng(0)

    % train one autoencoder per key (or reuse a cache), then use the
    % frozen encoders as backbones for the TopoART-AM head
    [ownerEncoder, ownerDecoder, objectEncoder, objectDecoder] = ...
        getAutoencoders(cacheFile, trainOwners, trainObjects, ...
        ownerSize, objectSize, latentDim, aeMaxEpochs);

    % construct the TopoART-AM head operating in the two latent spaces
    % (key1 = owner latent, key2 = object latent)
    tamLayer = topoARTAssociativeMemoryLayer(latentDim, latentDim, ...
        moduleNum, rho_a, Beta_sbm = betaSbm, Phi = phi, Tau = tau, ...
        Direction = 'key1->key2', Name = 'topoART_AM');

    % assemble the (owner, object) image pairs following trainMap and learn
    % their latent associations using the encoders as frozen backbones
    [ownerPairs, objectPairs] = buildPairs(trainOwners, trainObjects, ...
        trainMap, imagesPerItem);
    ds = combine(arrayDatastore(ownerPairs), arrayDatastore(objectPairs));

    % Present in a fixed dataset order (Shuffle = 'never'), as in
    % associateImagesExample; a fixed order lets the network settle so
    % training can stop early, and keeps runs reproducible.
    disp('Train TopoART-AM on the latent pairs (incremental)')
    tamLayer = trainTopoARTAM(ownerEncoder, objectEncoder, tamLayer, ds, ...
        MaxEpochs = tamMaxEpochs, MiniBatchSize = size(ownerPairs, 1), ...
        Shuffle = 'never');

    % recall the objects of a test owner (key1 -> key2)
    testOwner = loadSingleImage(fullfile(datasetPath, 'test', ...
        'owners', 'owner_'), 3, 3, 'png');
    recallAndShow('Recall: owner -> objects', tamLayer, 'key1->key2', ...
        ownerEncoder, objectDecoder, testOwner, objectSize, ...
        recallActThresh, maxRecalls, ...
        imageFile('TopoART_head_owner_to_objects.png'));

    % recall the owners of a test object (key2 -> key1)
    testObject = loadSingleImage(fullfile(datasetPath, 'test', ...
        'objects', 'object_'), 10, 4, 'png');
    recallAndShow('Recall: object -> owners', tamLayer, 'key2->key1', ...
        objectEncoder, ownerDecoder, testObject, ownerSize, ...
        recallActThresh, maxRecalls, ...
        imageFile('TopoART_head_object_to_owners.png'));

end

function [ownerEnc, ownerDec, objectEnc, objectDec] = getAutoencoders( ...
        cacheFile, trainOwners, trainObjects, ownerSize, objectSize, ...
        latentDim, maxEpochs)
%GETAUTOENCODERS - Load cached autoencoders or train and cache them
%   With an empty cacheFile the autoencoders are always trained. With a
%   path, a matching cache is loaded; otherwise the autoencoders are
%   trained and saved to cacheFile so the slow gradient descent can be
%   reused while the TopoART-AM head is re-tuned. Delete the file to force
%   retraining after changing the autoencoder architecture.

    if ~isempty(cacheFile) && isfile(cacheFile)
        cached = load(cacheFile);
        if isCacheValid(cached, ownerSize, objectSize, latentDim, ...
                maxEpochs)
            disp('Reusing cached autoencoders')
            ownerEnc = cached.ownerEncoder;
            ownerDec = cached.ownerDecoder;
            objectEnc = cached.objectEncoder;
            objectDec = cached.objectDecoder;
            return
        end
        warning(['Cached autoencoders do not match the current ' ...
            'settings; retraining.'])
    end

    disp('Pre-training owner autoencoder (gradient descent)')
    [ownerEnc, ownerDec] = trainAutoencoder( ...
        imagesToVectors(trainOwners), ownerSize, latentDim, maxEpochs);
    disp('Pre-training object autoencoder (gradient descent)')
    [objectEnc, objectDec] = trainAutoencoder( ...
        imagesToVectors(trainObjects), objectSize, latentDim, maxEpochs);

    if ~isempty(cacheFile)
        cache = struct( ...
            'ownerEncoder', ownerEnc, 'ownerDecoder', ownerDec, ...
            'objectEncoder', objectEnc, 'objectDecoder', objectDec, ...
            'ownerSize', ownerSize, 'objectSize', objectSize, ...
            'latentDim', latentDim, 'maxEpochs', maxEpochs); %#ok<NASGU>
        save(cacheFile, '-struct', 'cache')
        fprintf('Saved autoencoders to %s\n', cacheFile)
    end
end

function valid = isCacheValid(cached, ownerSize, objectSize, ...
        latentDim, maxEpochs)
%ISCACHEVALID - True if a loaded cache matches the current settings

    fields = {'ownerEncoder', 'ownerDecoder', 'objectEncoder', ...
        'objectDecoder', 'ownerSize', 'objectSize', 'latentDim', ...
        'maxEpochs'};
    valid = all(isfield(cached, fields)) ...
        && isequal(cached.ownerSize, ownerSize) ...
        && isequal(cached.objectSize, objectSize) ...
        && isequal(cached.latentDim, latentDim) ...
        && isequal(cached.maxEpochs, maxEpochs);
end

function [encoder, decoder] = trainAutoencoder(vecs, imgSize, ...
        latentDim, maxEpochs)
%TRAINAUTOENCODER - Train a convolutional autoencoder and split it
%   Trains a convolutional autoencoder on the [0, 1] image row vectors in
%   vecs (one observation per row, the flattened native image of size
%   imgSize) and returns the frozen encoder (image vector -> TopoART key
%   in [0, 1]) and decoder (key -> image vector in [0, 1]).
%
%   The images keep their native size: the encoder downsamples with
%   'same' padding and the decoder doubles the size back, trimming the
%   few surplus pixels with a final crop. The bottleneck is linear, so
%   internal values cannot saturate. After training, a per-dimension
%   affine rescale is fitted on the training latents and baked into both
%   halves: the encoder maps each latent into the sub-interval
%   [0.25, 0.75], which keeps the key inside the [0, 1] range TopoART
%   requires and leaves head room for latents that drift outside the
%   training range; the decoder inverts the same rescale before
%   reconstructing, so a recalled key is decoded without any extra
%   transform.

    imgH = imgSize(1);
    imgW = imgSize(2);
    flatLen = imgH * imgW * 3;

    % four stride-2 stages downsample the image; featChan is the channel
    % count of the deepest feature map and featH/featW its spatial size
    numStages = 4;
    featChan = 64;
    featH = downSize(imgH, numStages);
    featW = downSize(imgW, numStages);

    encNames = {'enc_in', 'enc_reshape', 'enc_conv1', 'enc_relu1', ...
        'enc_conv2', 'enc_relu2', 'enc_conv3', 'enc_relu3', ...
        'enc_conv4', 'enc_relu4', 'enc_latent'};
    decNames = {'dec_fc', 'dec_relu', 'dec_reshape', 'dec_tconv1', ...
        'dec_drelu1', 'dec_tconv2', 'dec_drelu2', 'dec_tconv3', ...
        'dec_drelu3', 'dec_tconv4', 'dec_sig', 'dec_crop', 'dec_out'};

    aeLayers = [
        featureInputLayer(flatLen, Name = 'enc_in')
        functionLayer(@(x) dlarray(reshape(stripdims(x), imgH, ...
            imgW, 3, size(x, 2)), 'SSCB'), Formattable = true, ...
            Name = 'enc_reshape')
        convolution2dLayer(3, 16, Padding = 'same', Stride = 2, ...
            Name = 'enc_conv1')
        reluLayer(Name = 'enc_relu1')
        convolution2dLayer(3, 32, Padding = 'same', Stride = 2, ...
            Name = 'enc_conv2')
        reluLayer(Name = 'enc_relu2')
        convolution2dLayer(3, 64, Padding = 'same', Stride = 2, ...
            Name = 'enc_conv3')
        reluLayer(Name = 'enc_relu3')
        convolution2dLayer(3, featChan, Padding = 'same', Stride = 2, ...
            Name = 'enc_conv4')
        reluLayer(Name = 'enc_relu4')
        fullyConnectedLayer(latentDim, Name = 'enc_latent')
        fullyConnectedLayer(featH * featW * featChan, Name = 'dec_fc')
        reluLayer(Name = 'dec_relu')
        functionLayer(@(x) dlarray(reshape(stripdims(x), featH, ...
            featW, featChan, size(x, 2)), 'SSCB'), ...
            Formattable = true, Name = 'dec_reshape')
        transposedConv2dLayer(3, 64, Stride = 2, Cropping = 'same', ...
            Name = 'dec_tconv1')
        reluLayer(Name = 'dec_drelu1')
        transposedConv2dLayer(3, 32, Stride = 2, Cropping = 'same', ...
            Name = 'dec_tconv2')
        reluLayer(Name = 'dec_drelu2')
        transposedConv2dLayer(3, 16, Stride = 2, Cropping = 'same', ...
            Name = 'dec_tconv3')
        reluLayer(Name = 'dec_drelu3')
        transposedConv2dLayer(3, 3, Stride = 2, Cropping = 'same', ...
            Name = 'dec_tconv4')
        sigmoidLayer(Name = 'dec_sig')
        functionLayer(@(x) x(1:imgH, 1:imgW, :, :), ...
            Formattable = true, Name = 'dec_crop')
        functionLayer(@(x) dlarray(reshape(stripdims(x), [], ...
            size(x, 4)), 'CB'), Formattable = true, Name = 'dec_out')
    ];

    options = trainingOptions('adam', ...
        MaxEpochs = maxEpochs, ...
        MiniBatchSize = min(size(vecs, 1), 64), ...
        Shuffle = 'every-epoch', ...
        InitialLearnRate = 1e-3, ...
        Verbose = false, ...
        OutputFcn = @trainnetDots, ...
        Plots = 'none');

    % reconstruction target equals the input
    ae = trainnet(vecs, vecs, aeLayers, 'mse', options);

    % encoder with a linear latent (image vector -> unbounded latent)
    encoder = initialize(removeLayers(ae, decNames));

    % fit a per-dimension affine rescale on the training latents so the
    % key lands inside [0.25, 0.75]
    latents = extractdata(predict(encoder, dlarray(vecs', 'CB')));
    lo = min(latents, [], 2);
    hi = max(latents, [], 2);
    span = hi - lo;
    span(span < 1e-6) = 1;          % guard constant dimensions
    scale = single(0.5 ./ span);
    offset = single(0.25 - scale .* lo);

    % bake the rescale into the encoder so its output is the TopoART key
    encoder = addLayers(encoder, ...
        functionLayer(@(x) scale .* x + offset, ...
        Formattable = true, Name = 'enc_affine'));
    encoder = connectLayers(encoder, 'enc_latent', 'enc_affine');
    encoder = initialize(encoder);

    % decoder: invert the same rescale, then reconstruct the image. The
    % trained decoder weights are kept by removeLayers.
    decoder = removeLayers(ae, encNames);
    decoder = addLayers(decoder, ...
        featureInputLayer(latentDim, Name = 'dec_in'));
    decoder = addLayers(decoder, ...
        functionLayer(@(x) (x - offset) ./ scale, ...
        Formattable = true, Name = 'dec_invaffine'));
    decoder = connectLayers(decoder, 'dec_in', 'dec_invaffine');
    decoder = connectLayers(decoder, 'dec_invaffine', 'dec_fc');
    decoder = initialize(decoder);
end

function n = downSize(n0, numStages)
%DOWNSIZE - Spatial size after numStages stride-2 'same' downsamples

    n = n0;
    for s = 1:numStages
        n = ceil(n / 2);
    end
end

function recallAndShow(figName, tamLayer, direction, encoder, decoder, ...
        stimulus, recallSize, thresh, maxRecalls, exportFile)
%RECALLANDSHOW - Recall all associations and decode them to images
%   Encodes the stimulus, recalls every associated latent in the given
%   direction (1-to-n), and decodes each recalled latent into an image of
%   size recallSize (the native size of the recalled key). If exportFile
%   is given and non-empty, the figure size is fixed and the finished
%   figure is exported to exportFile as a PNG image.

    narginchk(9, 10)
    nargoutchk(0, 0)

    if nargin < 10
        exportFile = '';
    end

    tamLayer.Direction = direction;

    z = encodeImage(encoder, stimulus);
    [recalled, activations] = tamLayer.recall(z, thresh);

    nShown = min(size(recalled, 2), maxRecalls);

    if isempty(exportFile)
        figure(Name = figName)
    else
        % fix the figure size so that exported images are identical
        % across screens
        figure(Name = figName, Position = [100 100 1200 250])
    end

    % place all images at a common scale so that the native size
    % relation between the stimulus and the recalled images is kept
    positions = layoutImageRow([size(stimulus); ...
        repmat(recallSize, nShown, 1)]);

    axes(Position = positions(1, :))
    imshow(stimulus)
    if nShown == 0
        captionBelow('stimulus (no association)')
    else
        captionBelow('stimulus')
    end

    for k = 1:nShown
        rec = predict(decoder, dlarray(single(recalled(:, k)), 'CB'));
        img = reshape(extractdata(rec), recallSize);
        axes(Position = positions(k + 1, :))
        imshow(img)
        captionBelow(sprintf('recall %d (%.3g)', k, activations(k)))
    end

    if ~isempty(exportFile)
        exportgraphics(gcf, exportFile, Resolution = 100)
    end
end

function z = encodeImage(encoder, img)
%ENCODEIMAGE - Latent code of a single image

    v = imageToVector(img);
    z = double(extractdata(predict(encoder, dlarray(v, 'CB'))));
    z = z(:);
end

function vecs = imagesToVectors(images)
%IMAGESTOVECTORS - Stack all images of a key as [0,1] row vectors
%   All images of a key share the same native size.

    vecs = zeros(numel(images), numel(images{1, 1}), 'single');
    k = 0;
    for item = 1:size(images, 1)
        for img = 1:size(images, 2)
            k = k + 1;
            vecs(k, :) = imageToVector(images{item, img})';
        end
    end
end

function [ownerPairs, objectPairs] = buildPairs(trainOwners, ...
        trainObjects, trainMap, imagesPerItem)
%BUILDPAIRS - Assemble (owner, object) image pairs as [0,1] row vectors
%   Image i of an owner is paired with image i of each associated object.
%   Owner and object rows may have different lengths (different native
%   sizes); each side feeds its own encoder inside trainTopoARTAM.

    pairNum = 0;
    for owner = 1:numel(trainMap)
        pairNum = pairNum + numel(trainMap{owner}) * imagesPerItem;
    end

    ownerPairs  = zeros(pairNum, numel(trainOwners{1, 1}), 'single');
    objectPairs = zeros(pairNum, numel(trainObjects{1, 1}), 'single');

    k = 0;
    for owner = 1:numel(trainMap)
        for object = trainMap{owner}
            for img = 1:imagesPerItem
                k = k + 1;
                ownerPairs(k, :) = ...
                    imageToVector(trainOwners{owner, img})';
                objectPairs(k, :) = ...
                    imageToVector(trainObjects{object, img})';
            end
        end
    end
end

function v = imageToVector(img)
%IMAGETOVECTOR - Flatten a native-size image to a [0, 1] column vector
%   No resizing is applied; the image keeps its native resolution.

    v = single(img(:)) / 255;
end
