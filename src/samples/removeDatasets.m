%REMOVEDATASETS - Delete the downloaded sample datasets
%   REMOVEDATASETS removes the data folder next to the samples and thus the
%   datasets downloaded by getDatasets. It can be run whenever the data is
%   no longer needed; getDatasets downloads it again on demand.
function removeDatasets()

    narginchk(0, 0)
    nargoutchk(0, 0)

    % locate the data folder next to the samples
    basePath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(basePath, 'data');

    if isfolder(dataPath)
        disp('Remove datasets')
        rmdir(dataPath, 's')
    else
        disp('No datasets to remove')
    end

end
