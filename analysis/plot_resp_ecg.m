function plot_resp_ecg(X, fsTarget, fixedY)
    arguments
        X (:,2) double % N x 2 array of [resp, ecg] signals at 2000 Hz
        fsTarget (1,1) double = 25
        fixedY (1,1) logical = false
    end

    % Optional flag to keep y-limits fixed for the full recording.
    if nargin < 3
        fixedY = false;
    end

    % X is N x 2 double: [resp, ecg]
    fs   = 2000;                % original sampling rate in Hz
    dsFactor = fs / fsTarget;   % downsampling factor
    dt   = 1/fs;                % sample interval in seconds
    N    = size(X,1);
    t    = (0:N-1).' * dt;      % time vector in seconds
    hResp25Data = decimate(X(:,2), dsFactor);
    t25 = (0:numel(hResp25Data)-1).' * (1/fsTarget);

    % Choose how much time to display at once (e.g. 5 seconds)
    winDur = 60;                 % seconds
    winSamples = round(winDur*fs);

    % Create UI figure and axes
    f = figure('Name','Resp + ECG browser','NumberTitle','off');
    ax2 = subplot(2,1,1,'Parent',f);
    ax3 = subplot(2,1,2,'Parent',f);

    % Initial index range
    idxStart = 1;
    idxEnd   = min(N, idxStart + winSamples - 1);

    % Initial plots (only downsampled respiratory and ECG)
    idxStart25 = floor((idxStart - 1) / dsFactor) + 1;
    idxEnd25 = min(numel(hResp25Data), floor((idxEnd - 1) / dsFactor) + 1);
    hResp25 = plot(ax2, t25(idxStart25:idxEnd25), hResp25Data(idxStart25:idxEnd25));
    ylabel(ax2,'Resp 25 Hz');
    title(ax2,'Downsampled respiratory signal (25 Hz)');

    hECG  = plot(ax3, t(idxStart:idxEnd), X(idxStart:idxEnd,1));
    ylabel(ax3,'ECG');
    xlabel(ax3,'Time [s]');
    title(ax3,'ECG signal');

    % If requested, lock y-limits to global ranges for the entire signal.
    if fixedY
        respLimits = localComputeYLimits(hResp25Data);
        ecgLimits  = localComputeYLimits(X(:,1));
        ylim(ax2, respLimits);
        ylim(ax3, ecgLimits);
    end

    % Slider range is in samples, not seconds
    s = uicontrol('Style','slider',...
                  'Units','normalized',...
                  'Position',[0.1 0.01 0.8 0.04],...
                  'Min',1,...
                  'Max',max(1, N-winSamples+1),...
                  'Value',1,...
                  'SliderStep',[winSamples/max(1,N-1) , 0.1],...
                  'Callback',@sliderCallback);

    % Make sure slider updates continuously while dragging
    addlistener(s,'Value','PostSet',@(src,evt) sliderCallback(s,[]));

    function sliderCallback(src,~)
        idxStart = round(get(src,'Value'));
        idxStart = max(1, min(idxStart, N-winSamples+1));
        idxEnd   = min(N, idxStart + winSamples - 1);
        idxStart25 = floor((idxStart - 1) / dsFactor) + 1;
        idxEnd25 = min(numel(hResp25Data), floor((idxEnd - 1) / dsFactor) + 1);

        % Update the plotted data only for current window
        set(hResp25, 'XData', t25(idxStart25:idxEnd25), 'YData', hResp25Data(idxStart25:idxEnd25));
        set(hECG,  'XData', t(idxStart:idxEnd), 'YData', X(idxStart:idxEnd,1));

        if fixedY
            % Keep precomputed full-signal y-limits.
            ylim(ax2, respLimits);
            ylim(ax3, ecgLimits);
        else
            % Recompute limits from the currently displayed data window.
            ylim(ax2, 'auto');
            ylim(ax3, 'auto');
        end

        % Use the same time limits (in seconds) for both axes
        xlim(ax2, [t(idxStart) t(idxEnd)]);
        xlim(ax3, [t(idxStart) t(idxEnd)]);
    end

    function limits = localComputeYLimits(y)
        yMin = min(y);
        yMax = max(y);
        if yMin == yMax
            % Prevent degenerate y-limits for flat signals.
            delta = max(1e-6, 0.01 * max(abs(yMin), 1));
            limits = [yMin - delta, yMax + delta];
        else
            limits = [yMin, yMax];
        end
    end
end
