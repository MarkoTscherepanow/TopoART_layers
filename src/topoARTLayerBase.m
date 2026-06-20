classdef (Abstract) topoARTLayerBase < nnet.layer.Layer
%TOPOARTLAYERBASE - Abstract base class for TopoART-based custom layers
%   TOPOARTLAYERBASE collects the state and contracts that are common to
%   all custom deep learning layers wrapping LibTopoART.Compatibility.
%   Concrete subclasses instantiate the appropriate TopoART network class
%   in their constructor and implement the methods learn and predict.
%
%   IMPORTANT: TopoART neural networks are NOT trained via gradient
%   descent. They use incremental, online (sample-by-sample) ART-style
%   learning. As a consequence, layers derived from this base cannot be
%   trained with trainnet in the standard way. Subclasses expose a learn
%   method for incremental training. See topoARTClassificationLayer for a
%   concrete example.
%
%   The wrapped .NET object is stored as a handle reference in the property
%   Network. Adding the layer to a dlnetwork copies the layer (value class
%   semantics), but the property still refers to the same .NET instance, so
%   training performed via learn remains visible during prediction.
%
%   Common properties
%     InputLen   - Length of the input vector (channels of the dlarray)
%     ModuleNum  - Number of TopoART modules used by the wrapped network
%     Rho_a      - Vigilance parameter of the first TopoART module
%     NetType    - Network type from LibTopoART.Compatibility.Network
%     IOType     - Interface type used only for input/output
%
%   Transient properties
%     Network    - Handle to the wrapped LibTopoART.Compatibility TopoART
%                  network instance, marked transient because .NET handles
%                  cannot be serialised by MATLAB's deep learning
%                  framework. Use save and load to persist and restore the
%                  network state independently of the layer wrapper.
%
%   Methods that subclasses must implement
%     learn(layer, X, T) - Online training step or epoch (The shape and
%                          element type of T are subclass-specific.)
%     predict(layer, X)  - Forward pass (The number of output channels is
%                          subclass-specific.)
%
%   Methods provided by this base class
%     save(layer, path)  - Persist the wrapped network's state to a
%                          binary file (the layer wrapper itself is not
%                          serialised).
%     layer = load(layer, path)
%                        - Replace the wrapped network with one read
%                          from a binary file produced by save and
%                          refresh the layer-side cached
%                          hyperparameters. Must be assigned back
%                          because the layer is a value class.
%     resetAdaptationState(layer)
%                        - Clear the tracked adaptation state before an
%                          epoch of incremental training.
%     hasPermanentAdaptation(layer) / hasCandidateAdaptation(layer)
%                        - Check for any permanent or candidate adaptation
%                          since the last resetAdaptationState call; the
%                          permanent check lets training stop early.
%     has<Permanent|Candidate><Node|Weight|Edge>Adaptation(layer)
%                        - Check for a specific kind of adaptation: a node,
%                          weight, or edge change of a permanent or
%                          candidate neuron.

    properties

        % InputLen - Length of the input vector
        InputLen

        % ModuleNum - Number of TopoART modules
        ModuleNum

        % Rho_a - Vigilance parameter of the first TopoART module
        Rho_a

        % NetType - Network type (LibTopoART.Compatibility.Network)
        NetType

        % IOType - Interface type used only for input/output
        % The MATLAB type of the data exchanged with the wrapped network.
        % Allowed values: '' (the default) uses the floating-point
        % interface type (double); 'uint8' suits image data, which is
        % then passed as integers in [0, 255].
        IOType (1, :) char = ''

        % Beta_sbm - Learning rate of the second-best matching neuron
        Beta_sbm

        % Phi - Threshold for promoting candidate neurons to permanent ones
        Phi

        % Tau - Number of presentations between candidate-neuron purges
        Tau

        % R - Radial extend parameter of Hypersphere TopoART
        % (positive scalar, empty for TopoART where the parameter does not
        % exist)
        R

    end

    properties (Constant, Hidden)

        IntType = 'int64'
        FPType = 'double'

    end

    properties (Transient)

        % Network - Handle to the wrapped LibTopoART.Compatibility
        % network instance (TopoART_i64d or TopoART_i64du8).
        Network

    end

    methods

        function layer = set.IOType(layer, value)
            % IOType selects the wrapped network class, so it cannot be
            % changed once that network exists; assigning the unchanged
            % value stays allowed. The cross-property Network read is
            % safe (and the MCSUP warning suppressed) because IOType is
            % only assigned before the network is constructed.
            networkExists = ~isempty(layer.Network); %#ok<MCSUP>
            if networkExists && ~strcmp(value, layer.IOType)
                error(['Cannot change IOType once the wrapped ' ...
                    'network exists. Set IOType before calling ' ...
                    'load on a default-constructed layer, or ' ...
                    'construct the layer with the desired ' ...
                    'IOType.'])
            end
            layer.IOType = value;
        end

        function save(layer, path)
        %SAVE - Persist the wrapped network state to a binary file
        %   SAVE(layer, path) writes the wrapped TopoART network's
        %   state (categories, edges, hyperparameters) to path using
        %   the .NET library's binary format. The layer wrapper
        %   itself is not serialised; reload via the load method.

            arguments
                layer
                path {mustBeTextScalar}
            end

            if isempty(layer.Network)
                error(['Cannot save before the wrapped network is ' ...
                    'constructed.'])
            end
            layer.Network.Save(path);
        end

        function layer = load(layer, path)
        %LOAD - Replace wrapped network with one loaded from disk
        %   layer = LOAD(layer, path) reads a binary network file
        %   produced by save and assigns the resulting .NET instance
        %   to layer.Network. The layer-side reflections of the
        %   network's hyperparameters (InputLen, ModuleNum, Rho_a,
        %   Beta_sbm, Phi, Tau, and R for Hypersphere variants) are
        %   refreshed from the loaded network. NetType is not
        %   refreshed because the wrapped .NET object does not
        %   expose it; the caller is responsible for loading a
        %   network compatible with the layer subclass's role
        %   (e.g. a TopoART-C variant for topoARTClassificationLayer).
        %   IOType is likewise honoured but not refreshed: it selects
        %   which network class reads the file, so set it before calling
        %   load to read a uint8 network into a default-constructed
        %   layer.
        %
        %   The return value must be assigned back to layer because
        %   the layer is a value class and the wrapped .NET handle
        %   is replaced by this method.

            arguments
                layer
                path {mustBeTextScalar}
            end

            topoARTLayerBase.ensureLibLoaded()
            if exist(path, 'file') ~= 2
                error('Network file not found: %s', path)
            end

            % The interface types select which network class reads the
            % file.
            networkClass = topoARTLayerBase.networkClassName( ...
                layer.IntType, layer.FPType, layer.IOType);
            layer.Network = ...
                feval(['LibTopoART.Compatibility.' networkClass], path);

            layer.ModuleNum = double(layer.Network.ModuleNum);
            layer.Rho_a     = double(layer.Network.Rho_a);
            layer.Beta_sbm  = double(layer.Network.Beta_sbm);
            layer.Phi       = double(layer.Network.Phi);
            layer.Tau       = double(layer.Network.Tau);

            % Refresh the length and category-specific properties. The
            % default reflects InputLen and the radial extend R; subclasses
            % whose wrapped network has a different input structure (e.g.
            % two key vectors) override refreshNetworkProperties.
            layer = layer.refreshNetworkProperties();
        end

        function resetAdaptationState(layer)
        %RESETADAPTATIONSTATE - Clear the tracked adaptation state
        %   Reset the recorded adaptation state, covering the node,
        %   weight, and edge changes of both permanent and candidate
        %   neurons. Call it before an epoch of incremental training; the
        %   has...Adaptation predicates then report which kinds of
        %   adaptation that epoch caused, for example to stop training
        %   once the permanent network has stabilised.

            if isempty(layer.Network)
                error(['Cannot reset the adaptation state before the ' ...
                    'wrapped network is constructed.'])
            end
            layer.Network.ResetAdaptationState();
        end

        function adapted = hasPermanentAdaptation(layer, epsilon)
        %HASPERMANENTADAPTATION - Check for any permanent adaptation
        %   Returns true if a permanent node/edge was added or a permanent
        %   weight changed by more than epsilon since the last
        %   resetAdaptationState call; lets multi-epoch training stop once
        %   it is false. (epsilon default: 0.001)

            arguments
                layer
                epsilon (1, 1) double {mustBePositive} = 0.001
            end

            mask = LibTopoART.AdaptationState ...
                .ANY_PERMANENT_ADAPTATION_MASK;
            adapted = layer.adaptationMatches(mask, epsilon);
        end

        function adapted = hasCandidateAdaptation(layer, epsilon)
        %HASCANDIDATEADAPTATION - Check for any candidate adaptation
        %   Returns true if a candidate node/edge was added or removed or
        %   a candidate weight changed by more than epsilon since the last
        %   resetAdaptationState call. (epsilon default: 0.001)

            arguments
                layer
                epsilon (1, 1) double {mustBePositive} = 0.001
            end

            mask = LibTopoART.AdaptationState ...
                .ANY_NONPERMANENT_ADAPTATION_MASK;
            adapted = layer.adaptationMatches(mask, epsilon);
        end

        function adapted = hasPermanentNodeAdaptation(layer)
        %HASPERMANENTNODEADAPTATION - Check for an added permanent node
        %   Returns true if a permanent node was added since the last
        %   resetAdaptationState call. (Permanent nodes are never removed.)

            mask = LibTopoART.AdaptationState.ADDED_PERMANENT_NODE;
            adapted = layer.adaptationMatches(mask);
        end

        function adapted = hasPermanentWeightAdaptation(layer, epsilon)
        %HASPERMANENTWEIGHTADAPTATION - Check for a permanent weight change
        %   Returns true if a permanent weight changed by more than epsilon
        %   since the last resetAdaptationState call. (default: 0.001)

            arguments
                layer
                epsilon (1, 1) double {mustBePositive} = 0.001
            end

            mask = LibTopoART.AdaptationState.ADAPTED_PERMANENT_WEIGHT;
            adapted = layer.adaptationMatches(mask, epsilon);
        end

        function adapted = hasPermanentEdgeAdaptation(layer)
        %HASPERMANENTEDGEADAPTATION - Check for an added permanent edge
        %   Returns true if a permanent edge was added since the last
        %   resetAdaptationState call. (Permanent edges are never removed.)

            mask = LibTopoART.AdaptationState.ADDED_PERMANENT_EDGE;
            adapted = layer.adaptationMatches(mask);
        end

        function adapted = hasCandidateNodeAdaptation(layer)
        %HASCANDIDATENODEADAPTATION - Check for a candidate node change
        %   Returns true if a candidate node was added or removed since
        %   the last resetAdaptationState call.

            mask = bitor( ...
                LibTopoART.AdaptationState.ADDED_NODE_CANDIDATE, ...
                LibTopoART.AdaptationState.REMOVED_NODE_CANDIDATE);
            adapted = layer.adaptationMatches(mask);
        end

        function adapted = hasCandidateWeightAdaptation(layer, epsilon)
        %HASCANDIDATEWEIGHTADAPTATION - Check for a candidate weight change
        %   Returns true if a candidate weight changed by more than epsilon
        %   since the last resetAdaptationState call. (default: 0.001)

            arguments
                layer
                epsilon (1, 1) double {mustBePositive} = 0.001
            end

            mask = LibTopoART.AdaptationState.ADAPTED_NONPERMANENT_WEIGHT;
            adapted = layer.adaptationMatches(mask, epsilon);
        end

        function adapted = hasCandidateEdgeAdaptation(layer)
        %HASCANDIDATEEDGEADAPTATION - Check for a candidate edge change
        %   Returns true if a candidate edge was added or removed since
        %   the last resetAdaptationState call.

            mask = bitor( ...
                LibTopoART.AdaptationState.ADDED_EDGE_CANDIDATE, ...
                LibTopoART.AdaptationState.REMOVED_EDGE_CANDIDATE);
            adapted = layer.adaptationMatches(mask);
        end

    end

    methods (Access = private)

        function adapted = adaptationMatches(layer, mask, epsilon)
        %ADAPTATIONMATCHES - Test the adaptation state against a flag mask
        %   Returns true if any flag in mask is set in the adaptation
        %   state recorded since the last resetAdaptationState call. When
        %   epsilon is given, GetAdaptationState uses it as the
        %   weight-change threshold; otherwise the library default
        %   applies. The threshold does not affect node and edge flags.

            if isempty(layer.Network)
                error(['Cannot query the adaptation state before the ' ...
                    'wrapped network is constructed.'])
            end

            if nargin < 3
                state = layer.Network.GetAdaptationState();
            else
                state = layer.Network.GetAdaptationState(epsilon);
            end
            adapted = bitand(state, mask) ~= ...
                LibTopoART.AdaptationState.NO_ADAPTATION;
        end

    end

    methods (Access = protected)

        function typeName = inputOutputType(layer)
        %INPUTOUTPUTTYPE - MATLAB type used for input/output
        %   The type is IOType when set, otherwise the floating-point
        %   interface type.

            typeName = layer.IOType;
            if isempty(typeName)
                typeName = layer.FPType;
            end
        end

        function layer = refreshNetworkProperties(layer)
        %REFRESHNETWORKPROPERTIES - Refresh variant-specific properties
        %   Called by load after the shared hyperparameters have been
        %   refreshed from the wrapped network. The default reflects
        %   InputLen and the Hypersphere radial extend R. Subclasses whose
        %   wrapped network has a different input structure override this.

            layer.InputLen = double(layer.Network.InputLen);

            % R is only readable on Hypersphere TopoART variants; the
            % .NET property throws InvalidCastException otherwise.
            try
                layer.R = double(layer.Network.R);
            catch
                layer.R = [];
            end
        end

    end

    methods (Abstract)

        %LEARN - Incremental (non-gradient) training of the wrapped network
        %   Subclasses validate the type and shape of T before forwarding
        %   to the appropriate .NET Learn overload. The .NET network is
        %   adapted through its handle reference.
        learn(layer, X, T)

    end

    methods (Static)

        function ensureLibLoaded()
        %ENSURELIBLOADED - Load .NET DLL on first use
        %   Locates the DLL relative to this base class file and loads
        %   it via NET.addAssembly. A persistent guard makes subsequent
        %   calls a no-op so subclass constructors can call this freely
        %   on every instantiation. Errors out with a clear message if
        %   .NET is unavailable or the DLL is missing (e.g. installLibs
        %   has not been run).

            persistent loaded
            if ~isempty(loaded)
                return
            end

            if ~(ispc || isunix)
                error('OS is not supported by LibTopoART.Compatibility.')
            end
            if ~NET.isNETSupported
                error('.NET is not supported on this MATLAB installation.')
            end

            srcDir  = fileparts(mfilename('fullpath'));
            dllPath = fullfile(srcDir, 'lib', 'LibTopoART.Compatibility.dll');
            if exist(dllPath, 'file') ~= 2
                error(['LibTopoART.Compatibility.dll was not found at %s. ' ...
                    'Run installLibs to install the .NET library.'], dllPath)
            end

            NET.addAssembly(dllPath);
            loaded = true;
        end

        function name = networkClassName(intType, fpType, ioType)
        %NETWORKCLASSNAME - Build the TopoART network class name
        %   Maps the three interface types to the short name of the
        %   matching LibTopoART.Compatibility network class, e.g.
        %   ('int64', 'double', 'uint8') -> 'TopoART_i64du8'. Unknown
        %   types raise an error; whether the resulting class exists
        %   is decided by the installed library at construction.

            switch intType
                case 'int64'
                    intCode = 'i64';
                otherwise
                    error(['Unsupported IntType ''%s''. ' ...
                        'Known: ''int64''.'], intType)
            end

            switch fpType
                case 'double'
                    floatCode = 'd';
                otherwise
                    error(['Unsupported FPType ''%s''. ' ...
                        'Known: ''double''.'], fpType)
            end

            if isempty(ioType)
                ioCode = '';
            else
                switch ioType
                    case 'uint8'
                        ioCode = 'u8';
                    otherwise
                        error(['Unsupported IOType ''%s''. ' ...
                            'Known: '''' (none), ''uint8''.'], ioType)
                end
            end

            name = ['TopoART_' intCode floatCode ioCode];
        end

    end

end