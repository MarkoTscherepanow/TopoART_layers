%GETDATASETS - Download and unpack the sample datasets
%   GETDATASETS downloads the ObjectsOwners_dataset archive from the
%   LibTopoART website and unpacks it into the data folder next to the
%   samples, where the sample functions expect it. The downloaded archive
%   is removed after unpacking. Run GETDATASETS once before the
%   image-association samples are used. removeDatasets deletes the
%   downloaded data again.
function getDatasets()

    narginchk(0, 0)
    nargoutchk(0, 0)

    % URL of the dataset archive
    datasetUrl = ['https://www.LibTopoART.eu/download/data/' ...
        'ObjectsOwners_dataset.zip'];

    % locate the data folder next to the samples
    basePath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(basePath, 'data');
    datasetPath = fullfile(dataPath, 'ObjectsOwners_dataset');

    % nothing to do if the dataset is already present
    if isfolder(datasetPath)
        disp('Dataset already present')
        return
    end

    % create the data folder if necessary
    if ~isfolder(dataPath)
        mkdir(dataPath)
    end

    % download the archive; remove it again on any exit path
    zipPath = fullfile(dataPath, 'ObjectsOwners_dataset.zip');
    zipCleanup = onCleanup(@() deleteIfPresent(zipPath)); %#ok<NASGU>

    % a generous timeout avoids premature failures on large downloads
    disp(['Download dataset from ' datasetUrl])
    websave(zipPath, datasetUrl, weboptions('Timeout', 300));

    % unpack the archive into the data folder
    disp('Unpack dataset')
    unzip(zipPath, dataPath);

    if ~isfolder(datasetPath)
        error(['The archive did not contain the expected folder ' ...
            'ObjectsOwners_dataset.'])
    end

end

function deleteIfPresent(path)
%DELETEIFPRESENT - Delete a file if it exists

    if isfile(path)
        delete(path)
    end

end
