%INSTALLLIBS - Download and install the required .NET libraries
%   This function downloads and installs LibTopoART.Compatibility.dll
%   and its dependencies into the lib folder next to this file. It
%   needs to be run once before LibTopoART.Compatibility.dll can be
%   used. If the libraries have been installed correctly, no further
%   calls to INSTALLLIBS are required.
function installLibs()

    narginchk(0, 0)
    nargoutchk(0, 0)

    % set the versions of the required libraries
    fSharpCoreVersion = '10.1.301';
    systemNumericsVectorsVersion = '4.6.1';
    libTopoARTVersion = '1.0.0';
    libTopoARTCompatibilityVersion = '0.7.0';

    % check .NET support
    if ~(ispc || isunix)
        error('OS is not supported')
    elseif ~NET.isNETSupported
        error('.NET is not supported')
    end

    % check .NET runtime
    if ispc

        if exist('dotnetenv', 'builtin') > 0
            if strcmp(dotnetenv().Runtime, 'framework')
                disp('Use .NET Framework')
                useNetFramework = true;
            else
                disp('Use .NET')
                useNetFramework = false;
            end
        else
            disp('Try to use .NET Framework without check')
            useNetFramework = true;
        end

        dotnetCmd = 'dotnet';

    else

        disp('Use .NET')
        useNetFramework = false;

        if ismac
            dotnetCmd = '/usr/local/share/dotnet/dotnet';
        else
            dotnetCmd = 'dotnet';
        end

    end

    % load .NET runtime by accessing the namespace System
    disp('Load .NET runtime')
    System.Environment.Version;

    if useNetFramework
        libTarget = 'net472';
    else

        netVersion = sscanf(dotnetenv().Version, '.NET %i.%i.%i');
        if length(netVersion) == 3
            if netVersion(1) >= 10
                libTarget = 'net10.0';
            else
                libTarget = 'netstandard2.1';
            end
        else
            libTarget = 'netstandard2.1';
        end

    end

    % set the search path for the helper functions; restore it via
    % onCleanup on normal exit, error, or Ctrl+C
    basePath = fileparts(mfilename('fullpath'));
    oldSearchPath = addpath([basePath filesep 'helpers' filesep]);
    searchPathCleanup = onCleanup(@() path(oldSearchPath)); %#ok<NASGU>

    % install into the folder of this file regardless of the current
    % folder. The current folder is state separate from the search
    % path, so it gets its own restore.
    oldFolder = cd(basePath);
    folderCleanup = onCleanup(@() cd(oldFolder));

    % create the folders 'download' and 'lib' if not present yet
    disp('Create folders')

    if ~exist([basePath filesep 'download'], 'dir')
        mkdir('download')
    end
    if ~exist([basePath filesep 'lib'], 'dir')
        mkdir('lib')
    end

    if useNetFramework

        % download nuget.exe
        disp('Download nuget.exe (if required)')
        if ~exist(['download' filesep 'nuget.exe'], 'file')
            websave(['download' filesep 'nuget.exe'], ...
                ['https://dist.nuget.org/win-x86-commandline/' ...
                'latest/nuget.exe'], weboptions('Timeout', 300));
        end

        % install FSharp.Core.dll
        disp(['Install FSharp.Core.dll ' fSharpCoreVersion])
        execConsoleCmd(['download\nuget.exe install FSharp.Core ' ...
            '-Version ' fSharpCoreVersion ' -DependencyVersion ' ...
            'Ignore -OutputDirectory download'])
        copyfile(['download\FSharp.Core.' fSharpCoreVersion ...
            '\lib\netstandard2.0\FSharp.Core.dll'], 'lib')

        % install System.Numerics.Vectors.dll
        disp(['Install System.Numerics.Vectors.dll ' ...
            systemNumericsVectorsVersion])
        execConsoleCmd(['download\nuget.exe install ' ...
            'System.Numerics.Vectors -Version ' ...
            systemNumericsVectorsVersion ' -DependencyVersion Ignore ' ...
            '-OutputDirectory download'])
        copyfile(['download\System.Numerics.Vectors.' ...
            systemNumericsVectorsVersion ...
            '\lib\net462\System.Numerics.Vectors.dll'], 'lib')

        % install LibTopoART.dll
        disp(['Install LibTopoART.dll ' libTopoARTVersion])
        execConsoleCmd(['download\nuget.exe install LibTopoART ' ...
            '-Version ' libTopoARTVersion ' -DependencyVersion ' ...
            'Ignore -OutputDirectory download'])
        copyfile(['download\LibTopoART.' libTopoARTVersion ...
            '\lib\' libTarget '\LibTopoART.dll'], 'lib')

        % install LibTopoART.Compatibility.dll
        disp(['Install LibTopoART.Compatibility.dll ' ...
            libTopoARTCompatibilityVersion])
        execConsoleCmd(['download\nuget.exe install ' ...
            'LibTopoART.Compatibility -Version ' ...
            libTopoARTCompatibilityVersion ' -DependencyVersion ' ...
            'Ignore -OutputDirectory download'])
        copyfile(['download\LibTopoART.Compatibility.' ...
            libTopoARTCompatibilityVersion '\lib\' libTarget ...
            '\LibTopoART.Compatibility.*'], 'lib')

    else

        % create helper project
        disp('Create helper project')
        execConsoleCmd([dotnetCmd ' new console -o download ' ...
            '-n Helper --force'])

        % add dependencies
        disp('Add dependencies')
        execConsoleCmd([dotnetCmd ' add download' filesep ...
            'Helper.csproj package FSharp.Core -v ' fSharpCoreVersion])
        execConsoleCmd([dotnetCmd ' add download' filesep ...
            'Helper.csproj package LibTopoART -v ' libTopoARTVersion])
        execConsoleCmd([dotnetCmd ' add download' filesep ...
            'Helper.csproj package LibTopoART.Compatibility -v ' ...
            libTopoARTCompatibilityVersion])

        % restore (triggers download)
        disp('Download libraries')
        execConsoleCmd([dotnetCmd ' restore download' filesep ...
            'Helper.csproj --packages download' filesep])

        % install dependencies
        disp('Install dependencies')
        copyfile(['download' filesep 'fsharp.core' filesep ...
            fSharpCoreVersion filesep 'lib' filesep 'netstandard2.0' ...
            filesep 'FSharp.Core.dll'], 'lib')
        copyfile(['download' filesep 'libtopoart' filesep ...
            libTopoARTVersion filesep 'lib' filesep libTarget filesep ...
            'LibTopoART.dll'], 'lib')
        copyfile(['download' filesep 'libtopoart.compatibility' ...
            filesep libTopoARTCompatibilityVersion filesep 'lib' ...
            filesep libTarget filesep 'LibTopoART.Compatibility.*'], ...
            'lib')

    end

end
