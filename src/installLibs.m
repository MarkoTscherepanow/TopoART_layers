%INSTALLLIBS - Download and install the required .NET libraries
%   This function downloads and installs LibTopoART.Compatibility.dll and
%   its dependencies into the current folder. It needs to be run once
%   before LibTopoART.Compatibility.dll can be used. If the libraries have
%   been installed correctly, no further calls of INSTALLLIBS are required.
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
        copyCmd = 'copy';

    else

        disp('Use .NET')
        useNetFramework = false;

        if ismac
            dotnetCmd = '/usr/local/share/dotnet/dotnet';
        else
            dotnetCmd = 'dotnet';
        end

        copyCmd = 'cp';

    end

    % load .NET runtime by accessing the namespace System
    System.Console.WriteLine('Load .NET runtime')

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

    % set path for helper functions; restore via onCleanup so the
    % user's path is reverted even on error or Ctrl+C
    basePath = fileparts(mfilename('fullpath'));
    oldPath = addpath([basePath filesep 'helpers' filesep]);
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % create the folders 'download' and 'lib'
    disp('Create folders')

    execConsoleCmd('mkdir download')
    execConsoleCmd('mkdir lib')

    if useNetFramework 
        
        % download nuget.exe
        disp('Download nuget.exe (if required)')
        if ~exist(['download' filesep 'nuget.exe'], 'file')
            execConsoleCmd(['curl https://dist.nuget.org/win-x86-commandline/latest/nuget.exe ' ...
                '--output download\nuget.exe'])
        end

        % install FSharp.Core.dll
        disp(['Install FSharp.Core.dll ' fSharpCoreVersion])
        execConsoleCmd(['download\nuget.exe install FSharp.Core -version ' fSharpCoreVersion ...
            ' -DependencyVersion Ignore -OutputDirectory download'])
        execConsoleCmd([copyCmd ' download\FSharp.Core.' fSharpCoreVersion ...
            '\lib\netstandard2.0\FSharp.Core.dll lib'])

        % install System.Numerics.Vectors.dll
        disp(['Install System.Numerics.Vectors.dll ' systemNumericsVectorsVersion])
        execConsoleCmd(['download\nuget install System.Numerics.Vectors -Version ' ...
            systemNumericsVectorsVersion ' -DependencyVersion Ignore ' ...
            '-OutputDirectory download'])
        execConsoleCmd([copyCmd ' download\System.Numerics.Vectors.' systemNumericsVectorsVersion ...
            '\lib\net462\System.Numerics.Vectors.dll lib'])

        % install LibTopoART.dll
        disp(['Install LibTopoART.dll ' libTopoARTVersion])
        execConsoleCmd(['download\nuget install LibTopoART -Version ' libTopoARTVersion ...
            ' -DependencyVersion Ignore -OutputDirectory download'])
        execConsoleCmd([copyCmd ' download\LibTopoART.' libTopoARTVersion ...
            '\lib\' libTarget '\LibTopoART.dll lib'])

        % install LibTopoART.Compatibility.dll
        disp(['Install LibTopoART.Compatibility.dll ' libTopoARTCompatibilityVersion])
        execConsoleCmd(['download\nuget install LibTopoART.Compatibility -Version ' ...
            libTopoARTCompatibilityVersion ' -DependencyVersion Ignore -OutputDirectory download'])
        execConsoleCmd([copyCmd ' download\LibTopoART.Compatibility.' ...
            libTopoARTCompatibilityVersion '\lib\' libTarget '\LibTopoART.Compatibility.* lib'])

    else

        % create helper project
        disp('Create helper project')
        execConsoleCmd([dotnetCmd ' new console -o download -n Helper'])

        % add dependencies
        disp('Add dependencies')
        execConsoleCmd([dotnetCmd ' add download' filesep 'Helper.csproj ' ...
            'package FSharp.Core -v ' fSharpCoreVersion])
        execConsoleCmd([dotnetCmd ' add download' filesep 'Helper.csproj ' ...
            'package LibTopoART -v ' libTopoARTVersion])
        execConsoleCmd([dotnetCmd ' add download' filesep 'Helper.csproj ' ...
            'package LibTopoART.Compatibility -v ' libTopoARTCompatibilityVersion])

         % restore (triggers download)
         disp('Download libraries')
         execConsoleCmd([dotnetCmd ' restore download' filesep 'Helper.csproj ' ...
             '--packages download' filesep])

         % install dependencies
         disp('Install dependencies')
         execConsoleCmd([copyCmd ' download' filesep 'fsharp.core' filesep ...
             fSharpCoreVersion filesep 'lib' filesep 'netstandard2.0' filesep ...
             'FSharp.Core.dll lib'])
         execConsoleCmd([copyCmd ' download' filesep 'libtopoart' filesep ...
             libTopoARTVersion filesep 'lib' filesep libTarget filesep ...
             'LibTopoART.dll lib'])
         execConsoleCmd([copyCmd ' download' filesep 'libtopoart.compatibility' ...
             filesep libTopoARTCompatibilityVersion filesep 'lib' filesep ...
             libTarget filesep 'LibTopoART.Compatibility.* lib'])

    end

end
