%EXECCONSOLECMD - Execute a system command
%   This function executes a system command and evaluates its outcome.
%
%   Syntax
%     EXECCONSOLECMD(cmd)
%     EXECCONSOLECMD(cmd, onlyWarn)
%
%   Input Arguments
%     cmd - Command to be executed
%     onlyWarn - If true, a failed command only issues a warning.
%       (default: false)
function execConsoleCmd(cmd, onlyWarn)

    narginchk(1, 2)
    nargoutchk(0, 0)

    if nargin < 2
        onlyWarn = false;
    end

    [status, result] = system(cmd);
    if status ~= 0
        if onlyWarn
            warning(result)
        else
            error(result)
        end
    end

end