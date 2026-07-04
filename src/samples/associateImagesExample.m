%ASSOCIATEIMAGESEXAMPLE - Bidirectional image association with TopoART-AM
%   ATTENTION: This function requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs (in the parent folder) and
%   getDatasets (in this folder) must be run before ASSOCIATEIMAGESEXAMPLE
%   can be used.
%
%   ASSOCIATEIMAGESEXAMPLE demonstrates the use of the custom deep learning
%   layer topoARTAssociativeMemoryLayer as a bidirectional associative
%   memory. No backbone is involved: the raw images themselves are the
%   keys. The dataset contains two kinds of images grouped into owners and
%   objects, with an m-to-n mapping between them: each owner can be
%   associated with several objects and an object can be shared by several
%   owners.
%
%   As TopoART neural networks are not trained via gradient descent, the
%   layer is trained explicitly through its method learn (incremental
%   learning). The owner image is the first key, the object image the
%   second. With the uint8 interface type, the pixels are passed as
%   integers in [0, 255]. After training, recall is demonstrated in both
%   directions by switching Direction on the layer:
%     - presenting a test owner recalls its associated objects, and
%     - presenting a test object recalls its associated owners.
%   Recall is a 1-to-n mapping, so recall can return several associated
%   keys; by default it is stopped early (recallActThresh) so that only
%   keys with high F3 activations are returned. Because the keys are the
%   raw images, each recalled key is reshaped back into an image
%   directly.
%
%   TopoART-AM does not just store the training images but learns internal
%   representations called categories from them. The images produced by
%   recall are created from these categories and usually combine
%   information from several training images.
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
%     ASSOCIATEIMAGESEXAMPLE
%     ASSOCIATEIMAGESEXAMPLE(recallActThresh)
%     ASSOCIATEIMAGESEXAMPLE(recallActThresh, maxRecalls)
%
%   Input Arguments
%     recallActThresh - Minimum F3 activation kept during recall
%       Recall is a 1-to-n mapping. Only associations with at least this
%       activation are returned; lowering it yields more (weaker) matches.
%       (range: [0, 1]; default: 0.97)
%     maxRecalls - Maximum number of recalled images shown per stimulus
%       (positive integer; default: 5)
function associateImagesExample(recallActThresh, maxRecalls)

    narginchk(0, 2)
    nargoutchk(0, 0)

    if nargin < 1
        recallActThresh = 0.97;
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

    % number of TopoART modules
    moduleNum = 1;

    % vigilance parameter of the first module
    rho_a = 0.8;

    % learning rate of the second best-matching neuron
    betaSbm = 0.8;

    % threshold for rendering candidate neurons permanent
    phi = 5;

    % maximum number of training epochs
    maxEpochs = 25;

    % number of owners/objects and training images per item
    ownerNum = 6;
    objectNum = 20;
    imagesPerItem = 20;

    % m-to-n mapping of owners to objects (objects may be shared)
    trainMap = { ...
        [1 2 3 9 17 19], ...        % owner 1
        [3 4 9 15 17], ...          % owner 2
        [1 3 7 10 11 12 16 17], ... % owner 3
        [3 5 17 18 20], ...         % owner 4
        [3 8 14 19 20], ...         % owner 5
        [3 5 10 13 16]};            % owner 6

    % learning steps between purges of candidate neurons
    tau = 136;

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

    ownerSize  = size(trainOwners{1, 1});
    objectSize = size(trainObjects{1, 1});
    key1Len = prod(ownerSize);    % owner image = first key
    key2Len = prod(objectSize);   % object image = second key

    % construct the TopoART-AM head (pixels passed as integers in [0, 255])
    tamLayer = topoARTAssociativeMemoryLayer(key1Len, key2Len, ...
        moduleNum, rho_a, Beta_sbm = betaSbm, Phi = phi, Tau = tau, ...
        IOType = 'uint8', Name = 'topoART_AM');

    % Present each (owner, object) image pair following trainMap, pairing
    % image i of an owner with image i of its associated objects. Each
    % epoch first resets the adaptation state; once a full epoch no longer
    % causes any permanent adaptation the network has stabilised and
    % training stops early.
    disp('Train TopoART-AM on the raw image pairs (incremental)')
    trainStart = tic;
    for iter = 1:maxEpochs
        tamLayer.resetAdaptationState();
        for owner = 1:numel(trainMap)
            for object = trainMap{owner}
                for img = 1:imagesPerItem
                    tamLayer.learn(trainOwners{owner, img}(:)', ...
                        trainObjects{object, img}(:)');
                end
            end
            fprintf('.')   % progress: one dot per owner
        end
        if ~tamLayer.hasPermanentAdaptation()
            break
        end
    end
    fprintf('\n')
    fprintf('Training finished after %d epoch(s) in %.1f s\n', ...
        iter, toc(trainStart))

    % recall the objects of a test owner (key1 -> key2)
    testOwner = loadSingleImage(fullfile(datasetPath, 'test', ...
        'owners', 'owner_'), 3, 3, 'png');
    recallAndShow('Recall: owner -> objects', tamLayer, 'key1->key2', ...
        testOwner, objectSize, recallActThresh, maxRecalls, ...
        imageFile('TopoART_owner_to_objects.png'));

    % recall the owners of a test object (key2 -> key1)
    testObject = loadSingleImage(fullfile(datasetPath, 'test', ...
        'objects', 'object_'), 20, 4, 'png');
    recallAndShow('Recall: object -> owners', tamLayer, 'key2->key1', ...
        testObject, ownerSize, recallActThresh, maxRecalls, ...
        imageFile('TopoART_object_to_owners.png'));

end

function recallAndShow(figName, tamLayer, direction, stimulus, ...
        recallSize, thresh, maxRecalls, exportFile)
%RECALLANDSHOW - Recall associated images and display them
%   Sets Direction on the head and recalls the keys associated with
%   the flattened stimulus (1-to-n), keeping those with activations at
%   or above thresh. Each recalled key is reshaped back to an image of
%   size recallSize. If exportFile is given and non-empty, the figure
%   size is fixed and the finished figure is exported to exportFile as
%   a PNG image.

    narginchk(7, 8)
    nargoutchk(0, 0)

    if nargin < 8
        exportFile = '';
    end

    tamLayer.Direction = direction;

    recallStart = tic;
    [recalled, activations] = tamLayer.recall(stimulus(:), thresh);
    fprintf('%s in %.1f s\n', figName, toc(recallStart))

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
        img = uint8(reshape(recalled(:, k), recallSize));
        axes(Position = positions(k + 1, :))
        imshow(img)
        captionBelow(sprintf('recall %d (%.3g)', k, activations(k)))
    end

    if ~isempty(exportFile)
        exportgraphics(gcf, exportFile, Resolution = 100)
    end
end
