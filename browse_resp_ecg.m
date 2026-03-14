function browse_resp_ecg(X)
    % X is N x 2 double: [resp, ecg]
    fs   = 1 / 0.0005;        % 2000 Hz
    dt   = 1/fs;
    N    = size(X,1);
    t    = (0:N-1).' * dt;    % time vector in seconds

    % Choose how much time to display at once (e.g. 5 seconds)
    winDur = 5;               % seconds
    winSamples = round(winDur*fs);

    % Create UI figure and axes
    f = figure('Name','Resp + ECG browser','NumberTitle','off');
    ax1 = subplot(2,1,1,'Parent',f);
    ax2 = subplot(2,1,2,'Parent',f);

    % Initial index range
    idxStart = 1;
    idxEnd   = min(N, idxStart + winSamples - 1);

    % Initial plots
    hResp = plot(ax1, t(idxStart:idxEnd), X(idxStart:idxEnd,2));
    ylabel(ax1,'Resp');
    title(ax1,'Respiratory signal');

    hECG  = plot(ax2, t(idxStart:idxEnd), X(idxStart:idxEnd,1));
    ylabel(ax2,'ECG');
    xlabel(ax2,'Time [s]');
    title(ax2,'ECG signal');

    linkaxes([ax1,ax2],'x');  % keep same time range on both axes

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

        % Update the plotted data only for current window
        set(hResp, 'XData', t(idxStart:idxEnd), 'YData', X(idxStart:idxEnd,2));
        set(hECG,  'XData', t(idxStart:idxEnd), 'YData', X(idxStart:idxEnd,1));

        % Optionally fix y-limits, or let MATLAB auto-scale
        % ylim(ax1, [min(X(:,1)) max(X(:,1))]);
        % ylim(ax2, [min(X(:,2)) max(X(:,2))]);

        xlim(ax1, [t(idxStart) t(idxEnd)]);
        xlim(ax2, [t(idxStart) t(idxEnd)]);
    end
end
