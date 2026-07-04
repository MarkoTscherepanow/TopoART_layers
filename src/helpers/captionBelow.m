%CAPTIONBELOW - Write a caption centred below the current image axes
%   CAPTIONBELOW places a caption underneath the axes created by imshow. A
%   text object is used instead of title so that the caption sits below the
%   image and is not clipped by the tight axes that imshow creates.
%
%   Syntax
%     CAPTIONBELOW(str)
%
%   Input Arguments
%     str - Caption text
function captionBelow(str)

    narginchk(1, 1)
    nargoutchk(0, 0)

    text(0.5, -0.04, str, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'Clipping', 'off')

end
