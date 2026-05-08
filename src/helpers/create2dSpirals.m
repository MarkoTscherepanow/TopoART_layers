%CREATE2DSPIRALS - Generate two intertwined spirals
%   CREATE2DSPIRALS is a helper function generating the data points and
%   the class labels of two intertwined spirals in a two-dimensional space.
%   The data points lie in both dimensions within the interval [-6.5, 6.5].
%
%   Syntax
%     spirals = CREATE2DSPIRALS
%
%   Output Arguments
%     spirals - Data points of the spirals (format: x, y, class label)
function spirals = create2dSpirals()

    narginchk(0, 0)
    nargoutchk(1, 1)

    spirals = zeros(194, 3);
    for i = 0 : 96
        phi = i / 16 * pi;
        r = (6.5 * (104 - i)) / 104;

        spirals(i + 1, 1) = r * cos(phi);
        spirals(i + 1, 2) = r * sin(phi);
        spirals(i + 1, 3) = 1; % spiral 1 (class label)

        spirals(i + 98, 1) = -r * cos(phi);
        spirals(i + 98, 2) = -r * sin(phi);
        spirals(i + 98, 3) = 2; % spiral 2 (class label)
    end

end