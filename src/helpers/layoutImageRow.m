%LAYOUTIMAGEROW - Lay out a row of images at a common scale
%   LAYOUTIMAGEROW computes one axes position per image for a single row
%   inside the current figure. All images share a common scale factor,
%   so their displayed sizes keep the native size relation. The row is
%   centred in the figure with the bottom edges of all axes aligned, and
%   space is reserved below the axes for captions (see captionBelow).
%   Images are not upscaled beyond their native size.
%
%   Syntax
%     positions = LAYOUTIMAGEROW(imgSizes)
%
%   Input Arguments
%     imgSizes - Native image sizes
%       Matrix with one image per row; the first two columns hold the
%       image height and width in pixels (further columns, such as the
%       channel number, are ignored).
%
%   Output Arguments
%     positions - Axes positions
%       One row [left bottom width height] per image, given in
%       'normalized' figure units for use as the Position property of
%       axes.
function positions = layoutImageRow(imgSizes)

    narginchk(1, 1)
    nargoutchk(0, 1)

    % margins around the row and gap between images (in pixels)
    sideMargin = 10;
    topMargin = 10;
    bottomMargin = 30;
    gap = 20;

    imageNum = size(imgSizes, 1);
    heights = imgSizes(:, 1);
    widths = imgSizes(:, 2);

    figPos = get(gcf, 'Position');
    figWidth = figPos(3);
    figHeight = figPos(4);

    % common scale factor (at most 1 so that images keep at most their
    % native size)
    availWidth = figWidth - 2 * sideMargin - (imageNum - 1) * gap;
    availHeight = figHeight - topMargin - bottomMargin;
    scale = min([availWidth / sum(widths), ...
        availHeight / max(heights), 1]);

    % centre the row in the figure
    left = (figWidth - scale * sum(widths) - (imageNum - 1) * gap) / 2;
    bottom = bottomMargin + (availHeight - scale * max(heights)) / 2;

    positions = zeros(imageNum, 4);
    for k = 1:imageNum
        positions(k, :) = [left, bottom, ...
            scale * widths(k), scale * heights(k)];
        left = left + scale * widths(k) + gap;
    end

    positions(:, [1 3]) = positions(:, [1 3]) / figWidth;
    positions(:, [2 4]) = positions(:, [2 4]) / figHeight;

end
