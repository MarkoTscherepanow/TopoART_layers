%PLOTSURFACE - Plot one surface over a grid in its own figure
%   PLOTSURFACE(figName, plotTitle, xg, yg, z, zLimits, surfColormap,
%   exportFile) opens a figure named figName and draws the surface z
%   over the grid (xg, yg), titled plotTitle. The z-axis and colour
%   limits are set to zLimits and the colormap to surfColormap, so
%   separate figures stay comparable. If exportFile is non-empty, the
%   figure size is fixed and the finished figure is exported to
%   exportFile as a PNG image.
%
%   PLOTSURFACE(__, samples) additionally overlays the rows of samples
%   (format: x, y, z) as black markers, so the distribution of the
%   training data stays visible. The markers are lifted slightly above
%   their z-values so they are not buried in the surface.
function plotSurface(figName, plotTitle, xg, yg, z, zLimits, ...
        surfColormap, exportFile, samples)

    narginchk(8, 9)
    nargoutchk(0, 0)

    if nargin < 9
        samples = [];
    end

    if isempty(exportFile)
        figure('Name', figName)
    else
        % fix the figure size so that exported images are identical
        % across screens
        figure('Name', figName, 'Position', [100 100 700 560])
    end

    surf(xg, yg, z)
    title(plotTitle)
    grid on
    set(gca, 'fontsize', 15)
    colormap(surfColormap)
    xlabel('x')
    ylabel('y')
    zlabel('z')
    axis([min(xg(:)) max(xg(:)) min(yg(:)) max(yg(:)) zLimits])
    caxis(zLimits)

    if ~isempty(samples)
        % white edges keep the markers visible on dark surface regions
        hold on
        zLift = 0.02 * (zLimits(2) - zLimits(1));
        plot3(samples(:, 1), samples(:, 2), samples(:, 3) + zLift, ...
            'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', ...
            'MarkerSize', 4)
    end

    if ~isempty(exportFile)
        print(gcf, exportFile, '-dpng', '-r100')
    end

end
