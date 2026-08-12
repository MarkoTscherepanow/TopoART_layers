%TRAINNETDOTS - Print a progress dot every 20 epochs during trainnet
%   Pass as OutputFcn to trainingOptions, together with Verbose = false:
%     trainingOptions(..., Verbose = false, OutputFcn = @TRAINNETDOTS)
%   Prints one dot per 20 completed epochs and a newline when training
%   finishes, providing lightweight progress output without the verbose
%   trainnet display.
function stop = trainnetDots(info)

    persistent lastDotEpoch
    stop = false;

    if strcmp(info.State, 'start')
        lastDotEpoch = 0;
    elseif strcmp(info.State, 'iteration')
        if info.Epoch - lastDotEpoch >= 20
            fprintf('.')
            lastDotEpoch = info.Epoch;
        end
    elseif strcmp(info.State, 'done')
        fprintf('\n')
    end

end
