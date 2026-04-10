function plot_resp_ecg(X)

    % X is N x 2 double: [resp, ecg]
    fs   = 2000;                % original sampling rate in Hz
    fs25 = 25;                  % downsampled respiratory sampling rate in Hz
    dsFactor = fs / fs25;       % 2000 -> 25 Hz downsampling factor
    dt   = 1/fs;                % sample interval in seconds
    N    = size(X,1);
    t    = (0:N-1).' * dt;      % time vector in seconds
    hResp25Data = decimate(X(:,2), dsFactor);
    t25 = (0:numel(hResp25Data)-1).' * (1/fs25);

    % Choose how much time to display at once (e.g. 5 seconds)
    winDur = 60;                 % seconds
    winSamples = round(winDur*fs);

    % Create UI figure and axes
    f = figure('Name','Resp + ECG browser','NumberTitle','off');
    ax1 = subplot(3,1,1,'Parent',f);
    ax2 = subplot(3,1,2,'Parent',f);
    ax3 = subplot(3,1,3,'Parent',f);

    % Initial index range
    idxStart = 1;
    idxEnd   = min(N, idxStart + winSamples - 1);

    % Initial plots
    hResp = plot(ax1, t(idxStart:idxEnd), X(idxStart:idxEnd,2));
    ylabel(ax1,'Resp');
    title(ax1,'Respiratory signal');

    idxStart25 = floor((idxStart - 1) / dsFactor) + 1;
    idxEnd25 = min(numel(hResp25Data), floor((idxEnd - 1) / dsFactor) + 1);
    hResp25 = plot(ax2, t25(idxStart25:idxEnd25), hResp25Data(idxStart25:idxEnd25));
    ylabel(ax2,'Resp 25 Hz');
    title(ax2,'Downsampled respiratory signal (25 Hz)');

    hECG  = plot(ax3, t(idxStart:idxEnd), X(idxStart:idxEnd,1));
    ylabel(ax3,'ECG');
    xlabel(ax3,'Time [s]');
    title(ax3,'ECG signal');

    linkaxes([ax1,ax2,ax3],'x');  % keep same time range on all axes

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
        set(hResp, 'XData', t(idxStart:idxEnd), 'YData', X(idxStart:idxEnd,2));
        set(hResp25, 'XData', t25(idxStart25:idxEnd25), 'YData', hResp25Data(idxStart25:idxEnd25));
        set(hECG,  'XData', t(idxStart:idxEnd), 'YData', X(idxStart:idxEnd,1));

        % Optionally fix y-limits, or let MATLAB auto-scale
        % ylim(ax1, [min(X(:,1)) max(X(:,1))]);
        % ylim(ax2, [min(X(:,2)) max(X(:,2))]);

        xlim(ax1, [t(idxStart) t(idxEnd)]);
        xlim(ax2, [t(idxStart) t(idxEnd)]);
        xlim(ax3, [t(idxStart) t(idxEnd)]);
    end
end
