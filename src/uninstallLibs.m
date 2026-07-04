%UNINSTALLLIBS - Remove the downloaded .NET libraries
%   This function removes the files downloaded and installed by
%   installLibs. It must be run directly after starting MATLAB before
%   the first usage of LibTopoART.Compatibility.dll.
function uninstallLibs()

    narginchk(0, 0)
    nargoutchk(0, 0)

    % check OS support
    if ~(ispc || isunix)
        error('OS is not supported')
    end

    % remove the folders 'download' and 'lib' next to this file
    disp('Cleanup')

    basePath = fileparts(mfilename('fullpath'));
    removeFolder(fullfile(basePath, 'download'))
    removeFolder(fullfile(basePath, 'lib'))

end

function removeFolder(folder)
%REMOVEFOLDER - Remove a folder and its contents, warning if it fails

    [status, message] = rmdir(folder, 's');
    if ~status
        warning('%s', message)
    end

end
