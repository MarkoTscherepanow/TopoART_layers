classdef (Abstract) topoARTLayerBase < nnet.layer.Layer
%TOPOARTLAYERBASE - Abstract base class for TopoART-based custom layers
%   TOPOARTLAYERBASE collects the state and contracts that are common to
%   all custom deep learning layers wrapping LibTopoART.Compatibility.
%   Concrete subclasses instantiate the appropriate TopoART_i64d variant in
%   their constructor and implement the methods learn and predict.
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
%
%   Transient properties
%     Network    - Handle to the wrapped LibTopoART.Compatibility.TopoART_i64d
%                  instance, marked transient because .NET handles cannot
%                  be serialised by MATLAB's deep learning framework. Use
%                  the methods save and load to persist and restore the
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

    properties

        % InputLen - Length of the input vector
        InputLen

        % ModuleNum - Number of TopoART modules
        ModuleNum

        % Rho_a - Vigilance parameter of the first TopoART module
        Rho_a

        % NetType - Network type (LibTopoART.Compatibility.Network)
        NetType

        % Beta_sbm - Learning rate of the second-best matching neuron
        Beta_sbm

        % Phi - Threshold for promoting candidate neurons to permanent ones
        Phi

        % Tau - Number of presentations between candidate-neuron purges
        Tau

        % R - Radial extent parameter of Hypersphere TopoART
        % (positive scalar, empty for TopoART where the parameter does not
        % exist.)
        R

    end

    properties (Dependent)

        % Nu - Maximum number of F2 neurons used for predictio
        % (can be changed between subsequent prediction calls without
        % rebuilding the layer or the dlnetwork)
        Nu

    end

    properties (Transient)

        % Network - Handle to the wrapped LibTopoART.Compatibility.TopoART_i64d
        % instance.
        Network

    end

    methods

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
            layer.Network.Nu = int64(value);
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

            layer.Network = ...
                LibTopoART.Compatibility.TopoART_i64d(path);

            layer.InputLen  = double(layer.Network.InputLen);
            layer.ModuleNum = double(layer.Network.ModuleNum);
            layer.Rho_a     = double(layer.Network.Rho_a);
            layer.Beta_sbm  = double(layer.Network.Beta_sbm);
            layer.Phi       = double(layer.Network.Phi);
            layer.Tau       = double(layer.Network.Tau);

            % R is only readable on Hypersphere TopoART variants; the
            % .NET property throws InvalidCastException otherwise
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

    end

end