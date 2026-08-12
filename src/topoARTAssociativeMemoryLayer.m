classdef topoARTAssociativeMemoryLayer < topoARTLayerBase
%TOPOARTASSOCIATIVEMEMORYLAYER - TopoART-AM associative memory layer
%   TOPOARTASSOCIATIVEMEMORYLAYER wraps a TopoART-AM neural network from
%   LibTopoART.Compatibility and exposes it as a custom MATLAB deep
%   learning layer that acts as a bidirectional associative memory. The
%   wrapped class is selected by IOType. The default gives TopoART_i64d;
%   IOType = 'uint8' gives TopoART_i64du8, whose uint8 input/output suits
%   image data.
%
%   TopoART-AM learns associations between pairs of key vectors (key1,
%   key2). The association is symmetric: after training, either key can be
%   presented to recall the other. Because a deep learning forward pass has
%   a fixed topology, the recall direction is held in the property
%   Direction. It can be changed between prediction calls without
%   rebuilding the layer or the dlnetwork. The recalled output
%   reconstructs the associated key (key2 when Direction is
%   'key1->key2', key1 when it is 'key2->key1'), with the same length
%   and value range as the vectors passed to learn for that key ([0, 1],
%   or [0, 255] with IOType 'uint8'). Recovering the original data needs
%   the inverse of the mapping that produced those key vectors, for
%   example the matching decoder or an inverse scaling. Switching
%   Direction selects the other key, so this inverse mapping differs by
%   direction and is applied downstream rather than fixed in the graph.
%   The layer therefore has a single active input (the presented key)
%   and two outputs (the recalled vector and a corresponding
%   activation). Recall is a 1-to-n mapping: predict returns only the
%   strongest association so that it fits a fixed-size network output,
%   while recall returns the associated keys in order of descending
%   F3 activation. It may return the complete set, but it is usually
%   stopped earlier via a minimum activation so that only strongly
%   associated keys are returned.
%
%   IMPORTANT: TopoART neural networks are NOT trained via gradient
%   descent. They use incremental, online (sample-by-sample) ART-style
%   learning. As a consequence, this layer cannot be trained with trainnet
%   in the standard way. Instead, train it explicitly by calling its learn
%   method (or via trainTopoARTAM for a dual-backbone setup) before
%   assembling the dlnetwork.
%
%   INPUT RANGE: Every element of both key vectors must lie in [0, 1].
%   Mapping inputs into an inner sub-interval such as [0.25, 0.75] is
%   recommended, so that inputs slightly outside the training distribution
%   at inference time still fall inside [0, 1]. The same applies to the
%   recalled output: recall produces values in [0, 1], so the original
%   data scale is restored by the inverse of the recalled side's scaling.
%   An affine scaling inverts exactly. With IOType 'uint8' the keys are
%   integers in [0, 255] (scaled to [0, 1] internally).
%
%   The wrapped .NET object is stored as a handle reference in the property
%   Network (inherited from topoARTLayerBase). See topoARTLayerBase for
%   details on the shared state and the value-class semantics. Use the
%   inherited save and load methods to persist and restore the wrapped
%   network independently of the layer wrapper.
%
%   ATTENTION: This layer requires .NET Framework 4.7.2 or higher, or
%   .NET 6.0 or higher. Furthermore, installLibs must be run before
%   it can be used.
%
%   The wrapped network computes on the CPU. Inputs residing on a GPU
%   are gathered automatically before they reach the .NET library, and
%   predict returns its outputs in the input's environment.
%
%   Syntax
%     layer = TOPOARTASSOCIATIVEMEMORYLAYER(key1Len, key2Len, ...
%               moduleNum, rho_a)
%     layer = TOPOARTASSOCIATIVEMEMORYLAYER(__, Name=name)
%     layer = TOPOARTASSOCIATIVEMEMORYLAYER()
%       Default constructor: returns an uninitialised layer with no
%       wrapped .NET network. Populate it via layer = layer.load(path)
%       before calling learn or predict.
%
%   Input Arguments
%     key1Len   - Length of the first key vector. It is the channel
%                 count of the predict input when Direction is
%                 'key1->key2' (key1 presented) and of the recalled
%                 output when Direction is 'key2->key1'. Every element
%                 must lie in [0, 1]; see the INPUT RANGE note above.
%     key2Len   - Length of the second key vector. It is the channel
%                 count of the predict input when Direction is
%                 'key2->key1' and of the recalled output when Direction
%                 is 'key1->key2'.
%     moduleNum - Number of TopoART modules (typical: 2)
%     rho_a     - Vigilance parameter of the first TopoART module
%                 (in [0, 1]; higher values yield finer categories).
%     Direction - Recall direction during prediction.
%                 'key1->key2' presents key1 and recalls key2;
%                 'key2->key1' presents key2 and recalls key1.
%                 (default: 'key1->key2')
%     IOType    - Optional interface type used only for input/output.
%                 When empty, input/output uses the floating-point
%                 interface type (double); set it to 'uint8' for uint8
%                 input/output. (default: '')
%     Name      - Layer name (default: 'topoART_AM')
%     Beta_sbm  - Learning rate of the second-best-matching neuron
%                 (range: [0, 1]; default: leave the .NET library's
%                 default unchanged)
%     Phi       - Threshold for rendering candidate neurons permanent
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%     Tau       - Learning steps between purges of candidate neurons
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%
%   Outputs (predict)
%     recalled   - The recalled key vector (output 'recalled').
%                  Its length is the length of the non-presented key. A
%                  zero vector signals that the stimulus matched no stored
%                  association.
%     activation - The activation of the F3 node used for recall (output
%                  'activation', one channel). It is meaningful only when
%                  recalled is non-zero: a failed recall returns a zero
%                  recalled vector together with activation 0, so a
%                  positive activation serves as a confidence threshold.
%
%   Methods
%     learn(key1, key2)  - Train the wrapped TopoART-AM network online on
%                          the key pairs in the rows of key1 (size
%                          sampleNum-by-Key1Len) and key2 (size
%                          sampleNum-by-Key2Len). Phases of training and
%                          recall may be mixed arbitrarily.
%     predict            - Recall the strongest association in the
%                          configured Direction, producing the
%                          outputs described above (top-1, for use in a
%                          dlnetwork).
%     [recalled, activations] = recall(stimulus)
%                        - Recall the associations for a single stimulus
%                          (recalledKeyLen-by-recallNum and
%                          1-by-recallNum), ordered by descending
%                          activation. Optional arguments keep only
%                          matches at or above a minimum activation and
%                          cap the result at the maxRecalls strongest.
%     save(path)         - Persist the wrapped network to a binary file
%                          (inherited from topoARTLayerBase)
%     layer = load(path) - Replace the wrapped network with one read
%                          from a binary file produced by save
%                          (inherited from topoARTLayerBase; must be
%                          assigned back due to value-class semantics)

    properties

        % Key1Len - Length of the first key vector
        Key1Len

        % Key2Len - Length of the second key vector
        Key2Len

        % Direction - Recall direction during prediction
        % 'key1->key2' presents key1 and recalls key2; 'key2->key1'
        % presents key2 and recalls key1. May be changed between
        % subsequent prediction calls.
        Direction (1, :) char = 'key1->key2'

    end

    methods

        function layer = topoARTAssociativeMemoryLayer(varargin)

            % ensure the .NET library is loaded before any LibTopoART.*
            % type is referenced (idempotent across constructor calls)
            topoARTLayerBase.ensureLibLoaded()

            % a single active input (the presented key) and two named
            % outputs (the recalled vector and its activation); setting
            % OutputNames also fixes NumOutputs to 2
            layer.NumInputs   = 1;
            layer.OutputNames = {'recalled', 'activation'};

            % default constructor: produce an uninitialised layer that the
            % caller is expected to populate via load(path) before training
            % or recall, no .NET network is allocated
            if nargin == 0
                layer.Name = 'topoART_AM';
                layer.Description = ...
                    'TopoART-AM associative memory (uninitialised)';
                return
            end

            layer = layer.initialise(varargin{:});

        end

        function layer = set.Direction(layer, value)
            % validatestring normalises the value and rejects anything
            % outside the two supported directions.
            layer.Direction = validatestring(value, ...
                {'key1->key2', 'key2->key1'});
        end

        function [recalled, activation] = predict(layer, X)
        %PREDICT - Recall the strongest association in the set direction
        %   X is the unformatted dlarray for the presented key (channels x
        %   batch). The number of channels must match the presented key
        %   length implied by Direction. The first output holds the
        %   recalled key vector (channels x batch), the second its
        %   activation (1 x batch). The dlnetwork propagates the input
        %   format ('CB') to both outputs automatically.
        %
        %   Recall is a 1-to-n mapping. predict keeps only the strongest
        %   association per sample so that the output has a fixed size;
        %   the method recall steps through all associations of a single
        %   stimulus, usually stopped early via its minActivation
        %   argument to keep only strong matches.

            if isempty(layer.Network)
                error('topoARTAssociativeMemoryLayer:uninitialised', ...
                    ['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before predict.'])
            end

            % cast to the network's input/output interface type (e.g.
            % uint8); this also selects the matching .NET recall
            % overload. gather moves GPU-resident inputs to the CPU,
            % where the wrapped network computes; the 'like' casts below
            % return the outputs in the input's environment.
            inputData = extractdata(X);
            inputs = cast(gather(inputData), layer.inputOutputType());
            sampleNum = size(inputs, 2);

            if size(inputs, 1) ~= layer.presentedKeyLen()
                error('topoARTAssociativeMemoryLayer:sizeMismatch', ...
                    ['Number of input channels (%d) does not match ' ...
                    'the presented key length (%d) for ' ...
                    'Direction ''%s''.'], size(inputs, 1), ...
                    layer.presentedKeyLen(), layer.Direction)
            end

            recalled   = zeros(layer.recalledKeyLen(), sampleNum);
            activation = zeros(1, sampleNum);

            for i = 1:sampleNum
                [recalledKey, recallActivation] = ...
                    layer.recallStimulus(inputs(:, i)', 0, 1);
                if ~isempty(recallActivation)
                    recalled(:, i) = recalledKey;
                    activation(i)  = recallActivation;
                end
            end

            recalled   = dlarray(cast(recalled, 'like', inputData));
            activation = dlarray(cast(activation, 'like', inputData));

        end

        function [recalled, activations] = recall(layer, stimulus, ...
                minActivation, maxRecalls)
        %RECALL - Recall associated keys for one stimulus (1-to-n)
        %   [recalled, activations] = RECALL(layer, stimulus) presents a
        %   single key (a vector of length presentedKeyLen, given as a
        %   numeric array or an unformatted dlarray) in the configured
        %   Direction and returns its associated keys. recalled is
        %   recalledKeyLen-by-recallNum and activations is
        %   1-by-recallNum, ordered by descending activation.
        %
        %   [...] = RECALL(layer, stimulus, minActivation) keeps only the
        %   associations whose F3 activation is at least minActivation.
        %   (range: [0, 1]; default: 0)
        %
        %   [...] = RECALL(layer, stimulus, minActivation, maxRecalls)
        %   additionally keeps at most the maxRecalls strongest
        %   associations. (a positive integer or inf; default: inf)

            arguments
                layer
                stimulus
                minActivation (1, 1) double ...
                    {mustBeInRange(minActivation, 0, 1)} = 0
                maxRecalls (1, 1) double = inf
            end

            % The value inf is allowed, so mustBeInteger cannot be used.
            if maxRecalls < 1 || maxRecalls ~= floor(maxRecalls)
                error(['topoARTAssociativeMemoryLayer:' ...
                    'invalidMaxRecalls'], ...
                    'maxRecalls must be a positive integer or inf.')
            end

            if isempty(layer.Network)
                error('topoARTAssociativeMemoryLayer:uninitialised', ...
                    ['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before recall.'])
            end

            if isa(stimulus, 'dlarray')
                stimulus = extractdata(stimulus);
            end
            % gather moves a GPU-resident stimulus to the CPU, where the
            % wrapped network computes
            stimulus = cast(gather(stimulus(:)'), layer.inputOutputType());

            if numel(stimulus) ~= layer.presentedKeyLen()
                error('topoARTAssociativeMemoryLayer:sizeMismatch', ...
                    ['Stimulus length (%d) does not match the ' ...
                    'presented key length (%d) for Direction ' ...
                    '''%s''.'], numel(stimulus), ...
                    layer.presentedKeyLen(), layer.Direction)
            end

            % recallStimulus returns the associations at or above
            % minActivation, ordered by descending activation and capped
            % at the maxRecalls strongest
            [recalled, activations] = layer.recallStimulus(stimulus, ...
                minActivation, maxRecalls);

        end

        function learn(layer, key1, key2)
        %LEARN - Incremental training of the wrapped TopoART-AM network
        %   LEARN(layer, key1, key2) presents the key pairs formed by
        %   the rows of key1 (size sampleNum-by-Key1Len) and key2
        %   (size sampleNum-by-Key2Len) to the wrapped TopoART-AM
        %   network. Learning is symmetric, so the order of the keys
        %   only fixes which side Direction refers to. key1 and key2
        %   may be of any numeric type; pass uint8 directly when IOType
        %   is 'uint8'.

            arguments
                layer
                key1 (:, :) {mustBeNumeric}
                key2 (:, :) {mustBeNumeric}
            end

            if isempty(layer.Network)
                error('topoARTAssociativeMemoryLayer:uninitialised', ...
                    ['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before learn.'])
            end

            if size(key1, 2) ~= layer.Key1Len
                error('topoARTAssociativeMemoryLayer:sizeMismatch', ...
                    ['Number of columns in key1 (%d) does not match ' ...
                    'Key1Len (%d).'], size(key1, 2), layer.Key1Len)
            end

            if size(key2, 2) ~= layer.Key2Len
                error('topoARTAssociativeMemoryLayer:sizeMismatch', ...
                    ['Number of columns in key2 (%d) does not match ' ...
                    'Key2Len (%d).'], size(key2, 2), layer.Key2Len)
            end

            if size(key1, 1) ~= size(key2, 1)
                error('topoARTAssociativeMemoryLayer:sizeMismatch', ...
                    ['Number of rows in key1 (%d) must match number ' ...
                    'of rows in key2 (%d).'], size(key1, 1), ...
                    size(key2, 1))
            end

            layer.learnVectorPairs(key1, key2);

        end

    end

    methods (Access = protected)

        function layer = refreshNetworkProperties(layer)
        %REFRESHNETWORKPROPERTIES - Refresh the key lengths after load
        %   Overrides the base implementation: a TopoART-AM network has two
        %   key vectors rather than a single input, and the radial extend R
        %   does not apply.

            layer.Key1Len = double(layer.Network.Key1Len);
            layer.Key2Len = double(layer.Network.Key2Len);
            layer.R = [];
        end

    end

    methods (Access = private)

        function layer = initialise(layer, key1Len, key2Len, ...
                moduleNum, rho_a, options)

            arguments
                layer
                key1Len (1, 1) {mustBeInteger, mustBePositive}
                key2Len (1, 1) {mustBeInteger, mustBePositive}
                moduleNum (1, 1) {mustBeInteger, mustBePositive}
                rho_a (1, 1) double {mustBeInRange(rho_a, 0, 1)}
                options.Name (1, :) char = 'topoART_AM'
                options.Direction (1, :) char = 'key1->key2'
                options.Beta_sbm = []
                options.Phi = []
                options.Tau = []
                options.IOType (1, :) char = ''
            end

            % NetType is a fixed constant rather than a user choice
            netType = LibTopoART.Compatibility.Network.Fast_TopoART_AM;

            networkClass = topoARTLayerBase.networkClassName( ...
                layer.IntType, layer.FPType, options.IOType);

            layer.Name = options.Name;
            layer.Description = sprintf( ...
                ['TopoART-AM associative memory (key1 %d-d, key2 %d-d, ' ...
                '%d modules, rho_a = %g, %s)'], key1Len, key2Len, ...
                moduleNum, rho_a, networkClass);

            layer.Key1Len = key1Len;
            layer.Key2Len = key2Len;
            layer.ModuleNum = moduleNum;
            layer.Rho_a = rho_a;
            layer.NetType = netType;
            layer.IOType = options.IOType;
            layer.Direction = options.Direction;

            % instantiate the wrapped .NET network using the two-key
            % constructor overload
            className = ['LibTopoART.Compatibility.' networkClass];
            intType   = layer.IntType;
            floatType = layer.FPType;

            try
                layer.Network = feval(className, ...
                    cast(key1Len, intType), ...
                    cast(key2Len, intType), ...
                    cast(moduleNum, intType), ...
                    cast(rho_a, floatType), netType);
            catch constructErr
                error(['topoARTAssociativeMemoryLayer:' ...
                    'networkUnavailable'], ...
                    ['Could not construct %s as a TopoART-AM network.' ...
                    '\n\nUnderlying error:\n%s'], networkClass, ...
                    constructErr.message)
            end

            % apply optional hyperparameters to the wrapped network before
            % any learning happens; values left empty keep the .NET default
            if ~isempty(options.Beta_sbm)
                mustBeScalarOrEmpty(options.Beta_sbm)
                mustBeInRange(options.Beta_sbm, 0, 1)
                layer.Network.Beta_sbm = cast(options.Beta_sbm, floatType);
            end

            if ~isempty(options.Phi)
                mustBeScalarOrEmpty(options.Phi)
                mustBeInteger(options.Phi)
                mustBePositive(options.Phi)
                layer.Network.Phi = cast(options.Phi, intType);
            end

            if ~isempty(options.Tau)
                mustBeScalarOrEmpty(options.Tau)
                mustBeInteger(options.Tau)
                mustBePositive(options.Tau)
                layer.Network.Tau = cast(options.Tau, intType);
            end

            % reflect the actual values (user-set or library default) on
            % the layer so they can be inspected without reaching into the
            % wrapped .NET object
            layer.Beta_sbm = double(layer.Network.Beta_sbm);
            layer.Phi = double(layer.Network.Phi);
            layer.Tau = double(layer.Network.Tau);

        end

        function [recalled, activations] = recallStimulus(layer, ...
                stimulus, minActivation, maxRecalls)
        %RECALLSTIMULUS - Recall associations for one cast stimulus
        %   stimulus is a 1-by-presentedKeyLen row already cast to the
        %   input/output type. Returns recalled
        %   (recalledKeyLen-by-recallNum) and activations
        %   (1-by-recallNum) for the recallNum recall steps, ordered
        %   by descending activation.
        %
        %   minActivation stops recall once an activation drops below it
        %   (default: 0). maxRecalls caps recallNum at the strongest
        %   maxRecalls associations (default: inf).

            arguments
                layer
                stimulus
                minActivation (1, 1) double = 0
                maxRecalls (1, 1) double = inf
            end

            if strcmp(layer.Direction, 'key2->key1')
                numF3Nodes = layer.Network.BeginRecallKey1(stimulus);
            else
                numF3Nodes = layer.Network.BeginRecallKey2(stimulus);
            end

            % release the recall state on every exit path (normal,
            % error, or Ctrl+C) so a failed step cannot leave the
            % wrapped network in a half-open recall
            cleanup = onCleanup(@() layer.Network.EndRecall()); %#ok<NASGU>

            nSteps      = min(double(numF3Nodes), maxRecalls);
            recalled    = zeros(layer.recalledKeyLen(), nSteps);
            activations = zeros(1, nSteps);

            count = 0;
            for s = 1:nSteps

                response = layer.Network.RecallStep();

                % stop here, releasing recall via the cleanup above: an
                % empty recallResult is a failed (terminal) step, and
                % because steps arrive by descending activation the
                % first one below minActivation also ends the useful
                % sequence.
                if isempty(response.recallResult) ...
                        || response.F3_activation < minActivation
                    break
                end

                count = count + 1;
                result = double(response.recallResult);
                recalled(:, count) = result(:);
                activations(count) = response.F3_activation;

            end

            recalled    = recalled(:, 1:count);
            activations = activations(1:count);

        end

        function n = presentedKeyLen(layer)
        %PRESENTEDKEYLEN - Length of the key presented during recall

            if strcmp(layer.Direction, 'key1->key2')
                n = layer.Key1Len;
            else
                n = layer.Key2Len;
            end

        end

        function n = recalledKeyLen(layer)
        %RECALLEDKEYLEN - Length of the key recalled during prediction

            if strcmp(layer.Direction, 'key1->key2')
                n = layer.Key2Len;
            else
                n = layer.Key1Len;
            end

        end

    end

end
