%TRAINTOPOARTAM - Train a TopoART-AM head on two frozen backbones
%   TRAINTOPOARTAM drives online (non-gradient) training of a
%   topoARTAssociativeMemoryLayer that learns associations between the
%   features of two already-trained, frozen backbone dlnetworks. It mirrors
%   trainTopoART but feeds a key pair per sample, one key from each
%   backbone, while honouring TopoART's learning semantics:
%     - Both backbones are run in inference mode; their parameters are NOT
%       updated. (Frozen backbones are a hard precondition because
%       TopoART-AM builds stable prototypes in the two feature spaces; if a
%       space shifts, the prototypes become invalid.) Because they are
%       frozen, every sample is encoded once before training and the
%       cached keys are reused across epochs rather than re-encoded. The
%       backbones may run on a GPU; the cached keys are gathered to the
%       CPU, where the head computes.
%     - The head's wrapped .NET network is mutated in place by calling its
%       learn method per minibatch. There is no loss function and no
%       gradient computation.
%     - MaxEpochs defaults to 1. Values greater than 1 are particularly
%       useful when the head has been configured with Beta_sbm > 0 (so the
%       second-best-matching neuron takes partial steps that benefit from
%       repeated presentation). With the default Beta_sbm, additional
%       epochs are largely idempotent. With StopWhenStable (the default),
%       training ends as soon as a full epoch causes no permanent
%       adaptation, so MaxEpochs acts as an upper bound.
%
%   PRECONDITION (input range): Every element of both feature tensors must
%   lie in [0, 1]. The recommended pattern is to map into an inner
%   sub-interval such as [0.25, 0.75] so that features at inference time
%   that drift slightly outside the training distribution still fall inside
%   [0, 1]. TRAINTOPOARTAM does NOT rescale features for you: the transform
%   must be part of each backbone so that training and recall apply the
%   identical mapping.
%
%   Unlike trainTopoART, this function returns the trained head rather than
%   an assembled dlnetwork: the recall direction is a runtime choice and a
%   given inference graph contains only the presented side's backbone.
%   Assemble that graph separately (presented backbone -> trained head).
%
%   Syntax
%     tamLayer = TRAINTOPOARTAM(backbone1, backbone2, tamLayer, ds)
%     tamLayer = TRAINTOPOARTAM(__, Name=Value)
%
%   Input Arguments
%     backbone1 - Frozen dlnetwork producing the first key. Must have
%                 a single output whose channel count matches
%                 tamLayer.Key1Len.
%     backbone2 - Frozen dlnetwork producing the second key. Must have
%                 a single output whose channel count matches
%                 tamLayer.Key2Len.
%     tamLayer  - A topoARTAssociativeMemoryLayer instance.
%     ds        - Datastore yielding (X1, X2) pairs feeding backbone1 and
%                 backbone2 respectively. The minibatchqueue is configured
%                 via MiniBatchFormat (default {'CB', 'CB'}).
%
%   Name-Value Arguments
%     MaxEpochs       - Maximum passes over the data; an upper bound when
%                       StopWhenStable is true. (default: 1)
%     MiniBatchSize   - Samples per learn call. (default: 128)
%     Shuffle         - 'every-epoch' (default), 'once', or 'never'.
%     MiniBatchFormat - Cell of format strings for X1 and X2. (default:
%                       {'CB', 'CB'})
%     MiniBatchFcn    - Function handle that turns the cell arrays read
%                       from the datastore into tensors matching
%                       MiniBatchFormat. The default assumes tabular data
%                       with one row per sample for both keys.
%     Verbose         - Print epoch/iteration progress. (default: true)
%     StopWhenStable  - Stop once a full epoch causes no permanent
%                       adaptation, i.e. no added permanent node or
%                       edge and no permanent weight change (see
%                       hasPermanentAdaptation); TopoART has then
%                       stabilised, like the manual loop in
%                       associateImagesExample. A consistent order
%                       (Shuffle 'once'/'never') helps it settle;
%                       'every-epoch' may keep adapting past MaxEpochs.
%                       (default: true)
%
%   Output Arguments
%     tamLayer - The trained topoARTAssociativeMemoryLayer (its wrapped
%               network was mutated in place during training).
function tamLayer = trainTopoARTAM(backbone1, backbone2, tamLayer, ds, ...
        options)

    arguments
        backbone1 (1, 1) dlnetwork
        backbone2 (1, 1) dlnetwork
        tamLayer  (1, 1) topoARTAssociativeMemoryLayer
        ds
        options.MaxEpochs       (1, 1) {mustBeInteger, mustBePositive} = 1
        options.MiniBatchSize   (1, 1) {mustBeInteger, mustBePositive} = 128
        options.Shuffle         (1, :) char ...
            {mustBeMember(options.Shuffle, ...
            {'every-epoch', 'once', 'never'})} = 'every-epoch'
        options.MiniBatchFormat (1, :) cell = {'CB', 'CB'}
        options.MiniBatchFcn    (1, 1) function_handle = @defaultRowsToCB
        options.Verbose         (1, 1) logical = true
        options.StopWhenStable  (1, 1) logical = true
    end

    % Each backbone must have a single output we can wire into a key.
    if numel(backbone1.OutputNames) ~= 1
        error('trainTopoARTAM:multipleOutputs', ...
            'backbone1 must have a single output. Found %d outputs.', ...
            numel(backbone1.OutputNames))
    end
    if numel(backbone2.OutputNames) ~= 1
        error('trainTopoARTAM:multipleOutputs', ...
            'backbone2 must have a single output. Found %d outputs.', ...
            numel(backbone2.OutputNames))
    end

    % removeLayers drops the Initialized flag even when all remaining
    % learnables still hold trained values; running initialize is a no-op
    % for parameters that are already set, so it is safe here.
    if ~backbone1.Initialized
        backbone1 = initialize(backbone1);
    end
    if ~backbone2.Initialized
        backbone2 = initialize(backbone2);
    end

    mbq = minibatchqueue(ds, 2, ...
        MiniBatchSize   = options.MiniBatchSize, ...
        MiniBatchFcn    = options.MiniBatchFcn, ...
        MiniBatchFormat = options.MiniBatchFormat);

    if options.Verbose
        fprintf(['trainTopoARTAM: %d epoch(s), MiniBatchSize = %d, ' ...
            'Shuffle = %s\n'], options.MaxEpochs, options.MiniBatchSize, ...
            options.Shuffle);
    end

    % Encode every sample once. The frozen backbones produce the same keys
    % every epoch, so they are run here (in MiniBatchSize chunks to bound
    % memory) and the cached keys are reused across epochs. predict builds
    % no autograd tape, so the backbones are not updated; gather moves
    % GPU-resident keys to the CPU, where the wrapped network computes.
    reset(mbq)
    keys1Chunks = {};
    keys2Chunks = {};
    while hasdata(mbq)
        [X1, X2] = next(mbq);
        keys1Chunks{end + 1} = double(gather(extractdata( ...
            predict(backbone1, X1)))); %#ok<AGROW>
        keys2Chunks{end + 1} = double(gather(extractdata( ...
            predict(backbone2, X2)))); %#ok<AGROW>
    end
    keys1 = cat(2, keys1Chunks{:});
    keys2 = cat(2, keys2Chunks{:});

    if size(keys1, 1) ~= tamLayer.Key1Len
        error('trainTopoARTAM:sizeMismatch', ...
            ['backbone1 output channel count (%d) does not match ' ...
            'tamLayer.Key1Len (%d).'], size(keys1, 1), tamLayer.Key1Len)
    end
    if size(keys2, 1) ~= tamLayer.Key2Len
        error('trainTopoARTAM:sizeMismatch', ...
            ['backbone2 output channel count (%d) does not match ' ...
            'tamLayer.Key2Len (%d).'], size(keys2, 1), tamLayer.Key2Len)
    end

    % Present the cached keys to the head. Each minibatch is one learn call
    % that mutates the wrapped .NET network in place. The keys lie in two
    % small feature spaces, so this loop is cheap compared with encoding.
    sampleNum = size(keys1, 2);
    if strcmp(options.Shuffle, 'once')
        order = randperm(sampleNum);
    else
        order = 1:sampleNum;
    end

    iter = 0;
    for epoch = 1:options.MaxEpochs

        if strcmp(options.Shuffle, 'every-epoch')
            order = randperm(sampleNum);
        end

        if options.StopWhenStable
            tamLayer.resetAdaptationState();
        end

        for first = 1:options.MiniBatchSize:sampleNum
            iter = iter + 1;
            last = min(first + options.MiniBatchSize - 1, sampleNum);
            idx = order(first:last);
            tamLayer.learn(keys1(:, idx)', keys2(:, idx)');
        end

        if options.Verbose
            fprintf(['  epoch %d/%d done, %d samples, total ' ...
                'iterations: %d\n'], epoch, options.MaxEpochs, ...
                sampleNum, iter);
        end

        % stop once a full epoch causes no permanent adaptation
        % (stabilised)
        if options.StopWhenStable && ~tamLayer.hasPermanentAdaptation()
            if options.Verbose
                fprintf('  stabilised; stopping after epoch %d\n', epoch)
            end
            break
        end

    end

end

function [X1b, X2b] = defaultRowsToCB(data1, data2)
%DEFAULTROWSTOCB - Default preprocess for tabular row-per-sample datastores
%   Stacks the cell arrays read from the datastore into B-row matrices and
%   transposes them to ('C', 'B') ordering expected by featureInputLayer.

    X1b = cat(1, data1{:})';
    X2b = cat(1, data2{:})';

end
