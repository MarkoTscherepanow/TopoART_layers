%CREATE2DMOONS - Generate two interleaved half-moons
%   CREATE2DMOONS is a helper function generating data points and class
%   labels of two half-moons in a two-dimensional space. The data points
%   lie approximately within the intervals [-1.3, 2.3] in x-direction and
%   [-0.8, 1.3] in y-direction (depending on noise).
%
%   Two-moons are markedly easier for shallow MLPs than the intertwined
%   spirals from create2dSpirals, which makes them a good fit for
%   demonstrations where the backbone training itself is not the focus.
%
%   Syntax
%     moons = CREATE2DMOONS
%     moons = CREATE2DMOONS(samplesPerClass)
%     moons = CREATE2DMOONS(samplesPerClass, noise)
%
%   Input Arguments
%     samplesPerClass - Number of samples per moon. (default: 150)
%     noise - Standard deviation of additive Gaussian noise.
%       (default: 0.075)
%
%   Output Arguments
%     moons - Data points of the moons (format: x, y, class label)
function moons = create2dMoons(samplesPerClass, noise)

    narginchk(0, 2)
    nargoutchk(1, 1)

    if nargin < 1
        samplesPerClass = 150;
    end

    if nargin < 2
        noise = 0.075;
    end

    % moon 1
    angles1 = rand(samplesPerClass, 1) * pi;
    x1 = cos(angles1) + noise * randn(samplesPerClass, 1);
    y1 = sin(angles1) + noise * randn(samplesPerClass, 1);

    % moon 2
    angles2 = rand(samplesPerClass, 1) * pi;
    x2 = 1 - cos(angles2) + noise * randn(samplesPerClass, 1);
    y2 = 0.5 - sin(angles2) + noise * randn(samplesPerClass, 1);

    moons = [ x1, y1, ones(samplesPerClass, 1); ...
              x2, y2, 2 * ones(samplesPerClass, 1)];

end