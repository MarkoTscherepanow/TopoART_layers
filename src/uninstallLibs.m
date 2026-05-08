%UNINSTALLLIBS - Remove the downloaded .NET libraries
%   This function removes the downloaded and installed files from the
%   current folder. It must be run directly after starting MATLAB before
%   the first usage of LibTopoART.Compatibility.dll.
function uninstallLibs()

    narginchk(0, 0)
    nargoutchk(0, 0)

    % check OS support
    if ~(ispc || isunix)
        error('OS is not supported')
    end

    % set path for helper functions; restore via onCleanup so the
    % user's path is reverted even on error or Ctrl+C
    basePath = fileparts(mfilename('fullpath'));
    oldPath = addpath([basePath filesep 'helpers' filesep]);
    pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>

    % remove the folders 'download' and 'lib'
    disp('Cleanup')

    if ispc
        execConsoleCmd('rmdir /Q /S download', true)
        execConsoleCmd('rmdir /Q /S lib', true)
    else
        execConsoleCmd('rm -rf download', true)
        execConsoleCmd('rm -rf lib', true)
    end

end