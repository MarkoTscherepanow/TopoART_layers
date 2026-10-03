classdef topoARTRegressionLayer < topoARTLayerBase
%TOPOARTREGRESSIONLAYER - TopoART-R-based regression layer
%   TOPOARTREGRESSIONLAYER wraps a TopoART-R neural network from
%   LibTopoART.Compatibility and exposes it as a custom MATLAB deep
%   learning layer. The wrapped class is selected by IOType. The default
%   gives TopoART_i64d; IOType = 'uint8' gives TopoART_i64du8, whose
%   uint8 input/output suits image data. It is intended to be used as a
%   regression head, either standalone (directly behind a
%   featureInputLayer) or after a backbone whose features are fed into
%   TopoART-R.
%
%   TopoART-R learns the relation between an input vector (the
%   independent variables) and an output vector (the dependent
%   variables). During prediction, only the input vector is presented
%   and the dependent variables are estimated from the learnt
%   categories: categories enclosing the input are combined or, if none
%   encloses it, the neighbourhood set N of the best-matching neurons
%   is used, whose maximum cardinality is limited by Nu.
%
%   IMPORTANT: TopoART neural networks are NOT trained via gradient
%   descent. They use incremental, online (sample-by-sample) ART-style
%   learning. As a consequence, this layer cannot be trained with trainnet
%   in the standard way. Instead, train it explicitly by calling its learn
%   method before assembling the dlnetwork (or in between calls to
%   predict).
%
%   INPUT RANGE: For TopoART-R, every element of the input vector and of
%   the output vector must lie in [0, 1]. Mapping the data into an inner
%   sub-interval such as [0.1, 0.9] is recommended, so that inputs
%   slightly outside the training distribution at inference time still
%   fall inside [0, 1]. The rescaling may be a fixed transform decided
%   before training or a data-derived calibration; make sure that
%   training and inference use the identical mapping. The same applies
%   to the predicted output: predictions lie in [0, 1], so the original
%   data scale is restored by the inverse of the output vector's
%   scaling. An affine scaling inverts exactly. With IOType 'uint8' the
%   wrapped network expects integer input and output in [0, 255] (scaled
%   to [0, 1] internally).
%
%   The wrapped .NET object is stored as a handle reference in the property
%   Network (inherited from topoARTLayerBase). See topoARTLayerBase for
%   details on the shared state and the value-class semantics. MATLAB's save
%   and load functions persist and restore the layer, including the wrapped
%   network, via .mat files. The inherited save and load methods persist and
%   restore the wrapped network alone, using the binary LibTopoART format.
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
%     layer = TOPOARTREGRESSIONLAYER(inputLen, outputLen, moduleNum, ...
%               rho_a)
%     layer = TOPOARTREGRESSIONLAYER(inputLen, outputLen, moduleNum, ...
%               rho_a, netType)
%     layer = TOPOARTREGRESSIONLAYER(__, Name=name)
%     layer = TOPOARTREGRESSIONLAYER()
%       Default constructor: returns an uninitialised layer with no
%       wrapped .NET network. Populate it via layer = layer.load(path)
%       before calling learn or predict.
%
%   Input Arguments
%     inputLen  - Length of the input vector (independent variables;
%                 channels of the dlarray input)
%                 Every element must lie in [0, 1]; see the INPUT RANGE
%                 note above.
%     outputLen - Length of the output vector (dependent variables;
%                 channels of the dlarray output)
%     moduleNum - Number of TopoART modules (typical: 2)
%     rho_a     - Vigilance parameter of the first TopoART module
%                 (in [0, 1]; higher values yield finer categories).
%     netType   - Network type, given as a string or char vector
%                 Allowed values: 'TopoART_R' (accurate but slow
%                 decimal computations) or 'Fast_TopoART_R'
%                 (accelerated integer-mapped computations with
%                 slightly reduced accuracy). (default:
%                 'Fast_TopoART_R')
%     IOType    - Optional interface type used only for input/output.
%                 When empty, input/output uses the fixed floating-point
%                 interface type (double); set it to 'uint8' for uint8
%                 input/output. Only 'Fast_TopoART_R' is available with
%                 IOType 'uint8'. (default: '')
%     Name      - Layer name (default: 'topoART_R')
%     Beta_sbm  - Learning rate of the second-best-matching neuron
%                 Beta_sbm controls partial adaptation of the
%                 second-best match. (range: [0, 1]; default: leave the
%                 .NET library's default unchanged)
%     Phi       - Threshold for rendering candidate neurons permanent
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%     Tau       - Learning steps between purges of candidate neurons
%                 (positive integer; default: leave the .NET library's
%                 default unchanged)
%     Nu        - Maximum cardinality of the neighbourhood set N used
%                 for prediction (In the original TopoART-R network, nu
%                 is fixed to 10, but task-specific values may improve
%                 the prediction accuracy.) After construction, Nu can
%                 be changed between subsequent prediction calls via
%                 the layer's Nu property (the setter propagates to the
%                 wrapped .NET network). (default: leave the .NET
%                 library's default unchanged)
%
%   Methods
%     learn(X, T)        - Train the wrapped TopoART-R network online on the
%                          rows of X (size sampleNum-by-InputLen) using the
%                          target outputs T (size sampleNum-by-OutputLen).
%                          Phases of training and prediction may be mixed
%                          arbitrarily.
%     predict            - Forward pass producing an OutputLen-channel
%                          output (the predicted dependent variables) in
%                          'CB' format. NaN signals that no prediction is
%                          possible yet.
%     save(path)         - Persist the wrapped network to a binary file
%                          (inherited from topoARTLayerBase)
%     layer = load(path) - Replace the wrapped network with one read from a
%                          binary file produced by save (inherited from
%                          topoARTLayerBase; must be assigned back due to
%                          value-class semantics)

    properties

        % OutputLen - Length of the output vector (dependent variables)
        OutputLen

    end

    properties (Dependent)

        % Nu - Maximum cardinality of the neighbourhood set N during
        % prediction (can be changed between subsequent prediction calls
        % without rebuilding the layer or the dlnetwork)
        Nu

    end

    methods

        function layer = topoARTRegressionLayer(varargin)

            % ensure the .NET library is loaded before any LibTopoART.*
            % type is referenced (idempotent across constructor calls)
            topoARTLayerBase.ensureLibLoaded()

            % default constructor: produce an uninitialised layer that
            % the caller is expected to populate via load(path) before
            % training or prediction, no .NET network is allocated
            if nargin == 0
                layer.Name = 'topoART_R';
                layer.Description = ...
                    'TopoART-R regression (uninitialised)';
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
                error('topoARTRegressionLayer:uninitialised', ...
                    ['Cannot set Nu before the wrapped network is ' ...
                    'constructed.'])
            end
            layer.Network.Nu = cast(value, layer.IntType);
        end

        function prediction = predict(layer, X)
        %PREDICT - Forward pass through the wrapped TopoART-R network
        %   X is the unformatted dlarray for the layer input (channels x
        %   batch, with C == InputLen). The returned unformatted dlarray has
        %   OutputLen channels per sample holding the predicted dependent
        %   variables in [0, 1] (or [0, 255] with IOType 'uint8'). All
        %   outputs are NaN as long as no prediction is possible, i.e. while
        %   the final TopoART module contains no neurons; NaN cannot be
        %   confused with a genuine prediction. The dlnetwork propagates the
        %   input format ('CB') to the output automatically.

            if isempty(layer.Network)
                error('topoARTRegressionLayer:uninitialised', ...
                    ['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before predict.'])
            end

            % cast to the network's input/output interface type (e.g.
            % uint8); this also selects the matching .NET Predict
            % overload. gather moves GPU-resident inputs to the CPU,
            % where the wrapped network computes; the 'like' cast below
            % returns the outputs in the input's environment.
            inputData = extractdata(X);
            inputs = cast(gather(inputData), layer.inputOutputType());
            sampleNum = size(inputs, 2);

            if size(inputs, 1) ~= layer.InputLen
                error('topoARTRegressionLayer:sizeMismatch', ...
                    ['Number of input channels (%d) does not ' ...
                    'match InputLen (%d).'], size(inputs, 1), ...
                    layer.InputLen)
            end

            outputs = NaN(layer.OutputLen, sampleNum);
            nodeNums = double(layer.Network.NodeNum);
            if nodeNums(end) > 0
                for i = 1:sampleNum
                    outputs(:, i) = ...
                        double(layer.Network.Predict(inputs(:, i)'));
                end
            end

            prediction = dlarray(cast(outputs, 'like', inputData));

        end

        function learn(layer, X, T)
        %LEARN - Incremental training of the wrapped TopoART-R network
        %   LEARN(layer, X, T) presents the rows of X (size
        %   sampleNum-by-InputLen) together with the target outputs in
        %   T (size sampleNum-by-OutputLen) to the wrapped TopoART-R
        %   network. X and T may be of any numeric type; pass uint8
        %   directly when IOType is 'uint8'.

            arguments
                layer
                X (:, :) {mustBeNumeric}
                T (:, :) {mustBeNumeric}
            end

            if isempty(layer.Network)
                error('topoARTRegressionLayer:uninitialised', ...
                    ['Layer is uninitialised. Construct with ' ...
                    'positional arguments or call ' ...
                    'LAYER = LAYER.LOAD(PATH) before learn.'])
            end

            if size(X, 2) ~= layer.InputLen
                error('topoARTRegressionLayer:sizeMismatch', ...
                    ['Number of columns in X (%d) does not match ' ...
                    'InputLen (%d).'], size(X, 2), layer.InputLen)
            end

            if size(T, 2) ~= layer.OutputLen
                error('topoARTRegressionLayer:sizeMismatch', ...
                    ['Number of columns in T (%d) does not match ' ...
                    'OutputLen (%d).'], size(T, 2), layer.OutputLen)
            end

            if size(X, 1) ~= size(T, 1)
                error('topoARTRegressionLayer:sizeMismatch', ...
                    ['Number of rows in X (%d) must match number ' ...
                    'of rows in T (%d).'], size(X, 1), size(T, 1))
            end

            layer.learnVectorPairs(X, T);

        end

    end

    methods (Access = protected)

        function layer = refreshNetworkProperties(layer)
        %REFRESHNETWORKPROPERTIES - Refresh the vector lengths after load
        %   Overrides the base implementation: the wrapped network's
        %   InputLen covers the concatenation of the input and output
        %   vectors, so the layer reflects I_len and D_len instead. The
        %   radial extend R does not apply to TopoART-R.

            layer.InputLen  = double(layer.Network.I_len);
            layer.OutputLen = double(layer.Network.D_len);
            layer.R = [];
        end

    end

    methods (Access = private)

        function layer = initialise(layer, inputLen, outputLen, ...
                moduleNum, rho_a, netType, options)

            arguments
                layer
                inputLen (1, 1) {mustBeInteger, mustBePositive}
                outputLen (1, 1) {mustBeInteger, mustBePositive}
                moduleNum (1, 1) {mustBeInteger, mustBePositive}
                rho_a (1, 1) double {mustBeInRange(rho_a, 0, 1)}
                netType {mustBeTextScalar} = 'Fast_TopoART_R'
                options.Name (1, :) char = 'topoART_R'
                options.Beta_sbm = []
                options.Phi = []
                options.Tau = []
                options.Nu = []
                options.IOType (1, :) char = ''
            end

            % Only TopoART regression networks are supported by this
            % layer.
            netTypeName = validatestring(netType, ...
                {'TopoART_R', 'Fast_TopoART_R'});
            netType = LibTopoART.Compatibility.Network.(netTypeName);

            % resolve (and validate) the network class
            networkClass = topoARTLayerBase.networkClassName( ...
                layer.IntType, layer.FPType, options.IOType);

            layer.Name = options.Name;
            layer.Description = sprintf( ...
                ['TopoART-R regression (%d-d input, %d-d output, ' ...
                '%d modules, rho_a = %g, %s)'], inputLen, outputLen, ...
                moduleNum, rho_a, networkClass);

            layer.InputLen  = inputLen;
            layer.OutputLen = outputLen;
            layer.ModuleNum = moduleNum;
            layer.Rho_a     = rho_a;
            layer.NetType   = netTypeName;
            layer.IOType    = options.IOType;

            % instantiate the wrapped .NET network using the constructor
            % overload taking the input and output vector lengths
            className = ['LibTopoART.Compatibility.' networkClass];
            intType   = layer.IntType;
            floatType = layer.FPType;

            try
                layer.Network = feval(className, ...
                    cast(inputLen, intType), ...
                    cast(outputLen, intType), ...
                    cast(moduleNum, intType), ...
                    cast(rho_a, floatType), netType);
            catch constructErr
                error('topoARTRegressionLayer:networkUnavailable', ...
                    ['Could not construct %s for netType ''%s''.' ...
                    '\n\nUnderlying error:\n' ...
                    '%s'], networkClass, netTypeName, ...
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
