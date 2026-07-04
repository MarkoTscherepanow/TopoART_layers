%LOADIMAGES - Load multiple images from the objects-owners dataset
%   LOADIMAGES is a helper function loading images used for training
%   a TopoART-AM neural network.
%
%   Syntax
%     images = LOADIMAGES(path, folderNum, imageNum)
%     images = LOADIMAGES(path, folderNum, imageNum, imageType)
%
%   Input Arguments
%     path - Path of the dataset images
%       The path must contain everything up to the indices of the
%       respective objects or owners, which will be inserted automatically.
%     folderNum - Number of objects or owners to be considered
%     imageNum - Number of images to be loaded
%     imageType - Image file extension without the leading dot, for
%       example 'jpg' or 'png' (default: 'jpg')
%
%   Output Arguments
%     images - Loaded images
function images = loadImages(path, folderNum, imageNum, imageType)

    narginchk(3, 4)
    nargoutchk(1, 1)

    if nargin < 4
        imageType = 'jpg';
    end

    images = cell(folderNum, imageNum);
    for f = 1:folderNum
        for i = 1:imageNum
            images{f, i} = imread([path num2str(f, '%02d') filesep ...
                'image_' num2str(i, '%02d') '.' imageType]);
        end
    end
end