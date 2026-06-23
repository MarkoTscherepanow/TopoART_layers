classdef topoARTClassificationLayer < topoARTLayerBase
%TOPOARTCLASSIFICATIONLAYER - TopoART-based classification layer
%   TOPOARTCLASSIFICATIONLAYER wraps a TopoART-C or Hypersphere TopoART-C
%   neural network from LibTopoART.Compatibility and exposes it as a
%   custom MATLAB deep learning layer. The wrapped class is selected by
%   IOType. The default gives TopoART_i64d; IOType = 'uint8' gives
%   TopoART_i64du8, whose uint8 input/output suits image data. It
%   is intended to be used as a classification head, either standalone
%   (directly behind a featureInputLayer) or after a backbone whose
%   features are fed into (Hypersphere) TopoART-C.
%
%   IMPORTANT: TopoART neural networks are NOT trained via gradient
%   descent. They use incremental, online (sample-by-sample) ART-style
%   learning. As a consequence, this layer cannot be trained with trainnet
%   in the standard way. Instead, train it explicitly by calling its learn
%   method before assembling the dlnetwork (or in between calls to
%   predict).
%
%   INPUT RANGE: For TopoART-C, every element of every input vector must
%   lie in [0, 1]. Mapping inputs into an inner sub-interval such as
%   [0.1, 0.9] is recommended, so that inputs slightly outside the
%   training distribution at inference time still fall inside [0, 1].
%   The rescaling may be a fixed transform decided before training
%   (e.g. sigmoidLayer, or functionLayer(@(x) 0.5 + 0.4*tanh(x))) or a
%   data-derived calibration. Make sure that training and inference use
%   the identical mapping. Hypersphere TopoART is less strict wrt. the
%   scaling interval. Larger intervals need to be reflected by larger
%   values of the radial extend parameter R. With IOType 'uint8' the
%   wrapped network expects integer input in [0, 255] (scaled to [0, 1]
%   internally).
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
%   Syntax
%     layer = TOPOARTCLASSIFICATIONLAYER(inputLen, moduleNum, rho_a)
%     layer = TOPOARTCLASSIFICATIONLAYER(inputLen, moduleNum, rho_a, netType)
%     layer = TOPOARTCLASSIFICATIONLAYER(__, Name=name)
%     layer = TOPOARTCLASSIFICATIONLAYER()
%       Default constructor: returns an uninitialised layer with no
%       wrapped .NET network. Populate it via layer = layer.load(path)
%       before calling learn or predict.
%
%   Input Arguments
%     inputLen  - Length of the input vector (channels of the dlarray)
%                 For TopoART, every input element
%                 must lie in [0, 1]; see the INPUT RANGE note above.
%     moduleNum - Number of TopoART modules (typical: 2)
%     rho_a     - Vigilance parameter of the first TopoART module
%                 (in [0, 1]; higher values yield finer categories).
%     netType   - Network type, given as a string or char vector
%                 Allowed values: 'TopoART_C', 'Fast_TopoART_C', or
%                 'Hypersphere_TopoART_C'. (default: 'Fast_TopoART_C')
%     IOType    - Optional interface type used only for input/output.
%                 When empty, input/output uses the fixed floating-point
%                 interface type (double); set it to 'uint8' for uint8
%                 input/output. (default: '')
%     Name      - Layer name (default: 'topoART_C')
%     Beta_sbm  - Learning rate of the second-best matching neuron
%                 Beta_sbm controls partial adaptation of the
%                 second-best match. (range: [0, 1]; default: leave the
%                 .NET library's default unchanged)
%     Phi       - Threshold for rendering candidate neurons permanent
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%     Tau       - Learning steps between purges of candidate neurons
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%     R         - Radial extend parameter of Hypersphere TopoART
%                 (positive scalar; Required when netType is
%                 'Hypersphere_TopoART_C'; rejected for TopoART; set at
%                 construction and stored as a layer property; cannot be
%                 changed afterwards)
%     Nu        - Maximum number of F2 neurons used for prediction
%                 After construction, Nu can be changed between subsequent
%                 prediction calls via the layer's Nu property (the
%                 setter propagates to the wrapped .NET network).
%                 (default: leave the .NET library's default unchanged)
%
%   Methods
%     learn(X, T)        - Train the wrapped TopoART-C network online on
%                          the rows of X (size N-by-inputLen) using class
%                          IDs T (size N-by-1, integer >= 0). Phases of
%                          training and prediction may be mixed
%                          arbitrarily.
%     predict            - Forward pass producing a 2-channel output
%                          [classID; confidence] in 'CB' format
%     save(path)         - Persist the wrapped network to a binary file
%                          (inherited from topoARTLayerBase)
%     layer = load(path) - Replace the wrapped network with one read
%                          from a binary file produced by save
%                          (inherited from topoARTLayerBase; must be
%                          assigned back due to value-class semantics)

    properties (Dependent)

        % Nu - Maximum number of F2 neurons used for prediction
        % (can be changed between subsequent prediction calls without
        % rebuilding the layer or the dlnetwork)
        Nu

    end

    methods

        function layer = topoARTClassificationLayer(varargin)

            % ensure the .NET library is loaded before any LibTopoART.*
            % type is referenced (idempotent across constructor calls)
            topoARTLayerBase.ensureLibLoaded()

            % default constructor: produce an uninitialised layer that
            % the caller is expected to populate via load(path) before
            % training or prediction, no .NET network is allocated
            if nargin == 0
                layer.Name = 'topoART_C';
                layer.Description = ...
                    'TopoART-C classifier (uninitialised)';
                return
            end

            layer = layer.initialise(varargin{:});
        end

        function value = get.Nu(layer)
            if isempty(layer.Network)
                value = [];
            else
                value = double(layer.Network.Nu);
            end
        end

        function layer = set.Nu(layer, value)
            mustBeScalarOrEmpty(value)
            if isempty(value)
                return
            end
            mustBeNumeric(value)
            mustBeInteger(value)
            mustBeNonnegative(value)
            if isempty(layer.Network)
                error(['Cannot set Nu before the wrapped network is ' ...
                    'constructed.'])
            end
            layer.Network.Nu = cast(value, layer.IntType);
        end

        function prediction = predict(layer, X)
        %PREDICT - Forward pass through the wrapped TopoART-C network
        %   X is the unformatted dlarray for the layer input (channels
        %   x batch, with C == InputLen). The returned unformatted
        %   dlarray has two channels per sample: the predicted class ID
        %   (channel 1) and the corresponding confidence in [0, 1]
        %   (channel 2). A class ID of 0 indicates that the sample lies
        %   outside any known category. The dlnetwork propagates the
        %   input format ('CB') to the output automatically.

            if isempty(layer.Network)
                error(['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before predict.'])
            end

            % cast to the network's input/output interface type (e.g.
            % uint8); this also selects the matching .NET Classify
            % overload
            inputs = cast(extractdata(X), layer.inputOutputType());
            nBatch = size(inputs, 2);
            mask = false(1, layer.InputLen);

            classIDs    = zeros(1, nBatch);
            confidences = zeros(1, nBatch);
            for i = 1:nBatch
                classification = ...
                    layer.Network.Classify(inputs(:, i)', mask);
                classIDs(i)    = double(classification.classID);
                confidences(i) = double(classification.confidence);
            end

            prediction = dlarray(cast([classIDs; confidences], ...
                'like', extractdata(X)));
        end

        function learn(layer, X, T)
        %LEARN - Incremental training of the wrapped TopoART-C network
        %   LEARN(layer, X, T) presents the rows of X (size
        %   N-by-InputLen) together with the class IDs in T (size
        %   N-by-1) to the wrapped TopoART-C network. X may be of any
        %   numeric type; pass uint8 directly when IOType is 'uint8'.

            arguments
                layer
                X (:, :) {mustBeNumeric}
                T (:, 1) double {mustBeInteger, mustBeNonnegative}
            end

            if isempty(layer.Network)
                error(['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before learn.'])
            end

            if size(X, 2) ~= layer.InputLen
                error(['Number of columns in X (%d) does not match ' ...
                    'InputLen (%d).'], size(X, 2), layer.InputLen)
            end
            if size(X, 1) ~= length(T)
                error(['Number of rows in X (%d) must match length ' ...
                    'of T (%d).'], size(X, 1), length(T))
            end

            % cast to the interface types so the correct .NET Learn
            % overload is selected: X to the input/output type and the
            % class IDs to the integer type
            layer.Network.Learn(cast(X, layer.inputOutputType()), ...
                cast(T, layer.IntType));
        end

    end

    methods (Access = private)

        function layer = initialise(layer, inputLen, moduleNum, ...
                rho_a, netType, options)

            arguments
                layer
                inputLen (1, 1) {mustBeInteger, mustBePositive}
                moduleNum (1, 1) {mustBeInteger, mustBePositive}
                rho_a (1, 1) double {mustBeInRange(rho_a, 0, 1)}
                % netType is a string / char so the .NET assembly stays
                % out of the default expression and can be loaded lazily
                % via topoARTLayerBase.ensureLibLoaded (called from the
                % constructor before delegating here).
                netType {mustBeTextScalar} = 'Fast_TopoART_C'
                options.Name (1, :) char = 'topoART_C'
                % Optional TopoART hyperparameters. Default [] keeps the
                % .NET library's default; supplying a value forwards it
                % to the wrapped network before any learning happens.
                options.Beta_sbm = []
                options.Phi = []
                options.Tau = []
                options.R = []
                options.Nu = []
                options.IOType (1, :) char = ''
            end

            % Only TopoART classifiers are supported by this layer.
            netTypeName = validatestring(netType, ...
                {'TopoART_C', 'Fast_TopoART_C', 'Hypersphere_TopoART_C'});
            netType = LibTopoART.Compatibility.Network.(netTypeName);
            isHypersphere = strcmp(netTypeName, 'Hypersphere_TopoART_C');

            % resolve (and validate) the network class
            networkClass = topoARTLayerBase.networkClassName( ...
                layer.IntType, layer.FPType, options.IOType);

            % R is meaningful only for Hypersphere TopoART-C; require it
            % there and reject it for TopoART-C.
            if isHypersphere

                if isempty(options.R)
                    error(['R is required when netType is ' ...
                        '''Hypersphere_TopoART_C''.'])
                end
                mustBeScalarOrEmpty(options.R)
                mustBeNumeric(options.R)
                mustBePositive(options.R)

            elseif ~isempty(options.R)
                error(['R is only meaningful when netType is ' ...
                    '''Hypersphere_TopoART_C''.'])
            end

            layer.Name = options.Name;
            layer.Description = sprintf( ...
                ['TopoART-C classifier (%d-d input, %d modules, ' ...
                'rho_a = %g, %s)'], inputLen, moduleNum, rho_a, ...
                networkClass);

            layer.InputLen  = inputLen;
            layer.ModuleNum = moduleNum;
            layer.Rho_a     = rho_a;
            layer.NetType   = netType;
            layer.IOType    = options.IOType;

            % instantiate the wrapped .NET network (Hypersphere TopoART-C
            % uses a different .NET overload taking the radial extend
            % R and a boolean (true = classification) instead of a netType
            % enum.)
            className = ['LibTopoART.Compatibility.' networkClass];
            intType   = layer.IntType;
            floatType = layer.FPType;

            try
                if isHypersphere
                    layer.Network = feval(className, ...
                        cast(inputLen, intType), ...
                        cast(moduleNum, intType), ...
                        cast(rho_a, floatType), ...
                        cast(options.R, floatType), true);
                else
                    layer.Network = feval(className, ...
                        cast(inputLen, intType), ...
                        cast(moduleNum, intType), ...
                        cast(rho_a, floatType), netType);
                end
            catch constructErr
                error('topoARTClassificationLayer:networkUnavailable', ...
                    ['Could not construct %s for netType ''%s''.' ...
                    '\n\nUnderlying error:\n' ...
                    '%s'], networkClass, netTypeName, ...
                    constructErr.message)
            end

            if isHypersphere
                layer.R = double(options.R);
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

            % Nu: the setter validates and writes through to
            % layer.Network.Nu (which is also where the dependent getter
            % reads from), so no explicit reflection step is needed.
            if ~isempty(options.Nu)
                layer.Nu = options.Nu;
            end

            % reflect the actual values (user-set or library default) on
            % the layer so they can be inspected without reaching into
            % the wrapped .NET object
            layer.Beta_sbm = double(layer.Network.Beta_sbm);
            layer.Phi = double(layer.Network.Phi);
            layer.Tau = double(layer.Network.Tau);
        end

    end

end