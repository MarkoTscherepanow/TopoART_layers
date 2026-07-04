%LOADSINGLEIMAGE - Load a single image from the objects-owners dataset
%   LOADSINGLEIMAGE is a helper function loading a single image used as a
%   stimulus to trigger the recall mechanism of a TopoART-AM neural
%   network.
%
%   Syntax
%     image = LOADSINGLEIMAGE(path, folderIndex, imageIndex)
%     image = LOADSINGLEIMAGE(path, folderIndex, imageIndex, imageType)
%
%   Input Arguments
%     path - Path of the dataset images
%       The path must contain everything up to the indices of the
%       respective objects or owners, which will be inserted automatically.
%     folderIndex - Index of the object or the owner to be used
%     imageIndex - Index of the image to be loaded
%     imageType - Image file extension without the leading dot, for
%       example 'jpg' or 'png' (default: 'jpg')
%
%   Output Arguments
%     image - Loaded image
function image = loadSingleImage(path, folderIndex, imageIndex, imageType)

    narginchk(3, 4)
    nargoutchk(1, 1)

    if nargin < 4
        imageType = 'jpg';
    end

    image = imread([path num2str(folderIndex, '%02d') filesep 'image_' ...
        num2str(imageIndex, '%02d') '.' imageType]);
end