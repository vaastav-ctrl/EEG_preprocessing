clc;
clear;

% Loop through the participants
for participant = 1:1
    % Loop through the runs for each participant
    for run = 3:3
        % Construct the filename for each run
        filename = sprintf('/Users/vaastav/Desktop/ncan/motor imagery/data/files/preprocessed/preprocessed_data_participant%d_run%d.set', participant, run);

        % Load the preprocessed dataset
        EEG = pop_loadset('filename', filename);

        % Calculate epoch duration in data points
        samplingRate = EEG.srate; % Sampling rate from the loaded dataset
        windowDurationInSeconds = 4.1; % Desired duration of each epoch window in seconds
        epochDuration = round(samplingRate * windowDurationInSeconds);

        % Get the event annotations
        annotations = {EEG.event.type};

        % Initialize arrays to store the epochs for T0, T1, T2
        epochs_T0 = [];
        epochs_T1_left = [];
        epochs_T1_both = [];
        epochs_T2_right = [];
        epochs_T2_both = [];

        % Extract epochs based on event types
        for j = 1:numel(annotations)
            eventType = annotations{j};

            switch eventType
                case 'T0'
                    % Extract epoch for T0
                    epochs_T0 = [epochs_T0; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds], 'newname', 'T0', 'epochinfo', 'yes')];
                case 'T1'
                    % Check the run number to determine if it's left fist or both fists
                    if ismember(run, [3, 4, 7, 8, 11, 12])
                        % Extract epoch for left fist (T1)
                        epochs_T1_left = [epochs_T1_left; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds], 'newname', 'T1 left', 'epochinfo', 'yes')];
                    else
                        % Extract epoch for both fists (T1)
                        epochs_T1_both = [epochs_T1_both; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds], 'newname', 'T1 both', 'epochinfo', 'yes')];
                    end
                case 'T2'
                    % Check the run number to determine if it's right fist or both feet
                    if ismember(run, [3, 4, 7, 8, 11, 12])
                        % Extract epoch for right fist (T2)
                        epochs_T2_right = [epochs_T2_right; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds], 'newname', 'T2 right', 'epochinfo', 'yes')];
                    else
                        % Extract epoch for both feet (T2)
                        epochs_T2_both = [epochs_T2_both; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds], 'newname', 'T2 both', 'epochinfo', 'yes')];
                    end
            end
        end

        % Save the epochs for each event separately
        save(sprintf('epochs_participant%d_run%d.mat', participant, run), 'epochs_T0', 'epochs_T1_left', 'epochs_T1_both', 'epochs_T2_right', 'epochs_T2_both');
    end 
end

% End of extracting epochs into arrays for each condition

%% Calculate and merge epochs

% Initialize a new struct to hold the merged data for T0
mergedEpochsT0 = struct();

% Loop through the fields of the first element and initialize the mergedEpochsT0 struct
fields = fieldnames(epochs_T0(1));
for i = 1:numel(fields)
    fieldName = fields{i};
    mergedEpochsT0.(fieldName) = epochs_T0(1).(fieldName);
end

% Loop through the other elements and merge their data into the mergedEpochsT0 struct
for elem = 2:numel(epochs_T0)
    for i = 1:numel(fields)
        fieldName = fields{i};
        mergedEpochsT0.(fieldName) = cat(3, mergedEpochsT0.(fieldName), epochs_T0(elem).(fieldName));
    end
end

% Initialize a new struct to hold the merged data for T1 left
mergedEpochsT1_left = struct();

% Loop through the fields of the first element and initialize the mergedEpochsT1_left struct
fields = fieldnames(epochs_T1_left(1));
for i = 1:numel(fields)
    fieldName = fields{i};
    mergedEpochsT1_left.(fieldName) = epochs_T1_left(1).(fieldName);
end

% Loop through the other elements and merge their data into the mergedEpochsT1_left struct
for elem = 2:numel(epochs_T1_left)
    for i = 1:numel(fields)
        fieldName = fields{i};
        mergedEpochsT1_left.(fieldName) = cat(3, mergedEpochsT1_left.(fieldName), epochs_T1_left(elem).(fieldName));
    end
end

% ... Repeat the above merging process for other conditions (T1 both, T2 right, T2 both)

% Initialize a new struct to hold the merged data for T1 both
mergedEpochsT1_both = struct();

% % Loop through the fields of the first element and initialize the mergedEpochsT1_both struct
% fields = fieldnames(epochs_T1_both(1));
% for i = 1:numel(fields)
%     fieldName = fields{i};
%     mergedEpochsT1_both.(fieldName) = epochs_T1_both(1).(fieldName);
% end
% 
% % Loop through the other elements and merge their data into the mergedEpochsT1_both struct
% for elem = 2:numel(epochs_T1_both)
%     for i = 1:numel(fields)
%         fieldName = fields{i};
%         mergedEpochsT1_both.(fieldName) = cat(3, mergedEpochsT1_both.(fieldName), epochs_T1_both(elem).(fieldName));
%     end
% end

% Initialize a new struct to hold the merged data for T2 right
mergedEpochsT2_right = struct();

% Loop through the fields of the first element and initialize the mergedEpochsT2_right struct
fields = fieldnames(epochs_T2_right(1));
for i = 1:numel(fields)
    fieldName = fields{i};
    mergedEpochsT2_right.(fieldName) = epochs_T2_right(1).(fieldName);
end

% Loop through the other elements and merge their data into the mergedEpochsT2_right struct
for elem = 2:numel(epochs_T2_right)
    for i = 1:numel(fields)
        fieldName = fields{i};
        mergedEpochsT2_right.(fieldName) = cat(3, mergedEpochsT2_right.(fieldName), epochs_T2_right(elem).(fieldName));
    end
end

% % Initialize a new struct to hold the merged data for T2 both
% mergedEpochsT2_both = struct();
% 
% % Loop through the fields of the first element and initialize the mergedEpochsT2_both struct
% fields = fieldnames(epochs_T2_both(1));
% for i = 1:numel(fields)
%     fieldName = fields{i};
%     mergedEpochsT2_both.(fieldName) = epochs_T2_both(1).(fieldName);
% end
% 
% % Loop through the other elements and merge their data into the mergedData struct
% for elem = 2:numel(epochs_T2_both)
%     for i = 1:numel(fields)
%         fieldName = fields{i};
%         mergedEpochsT2_both.(fieldName) = cat(3, mergedEpochsT2_both.(fieldName), epochs_T2_both(elem).(fieldName));
%     end
% end

%% Parameter Definitions

% Define spectral analysis parameters
spectralStep = 166;
spectralSize = 333;
samplefreq = EEG.srate;
modelOrd = 18 + round(samplefreq / 100);
hpCutoff = -1;
freqBinWidth = 4;
lp_cutoff = 71;
maxFreqEEG = 71;
trend = 1;
spectral_size = round(spectralSize / 1000 * samplefreq);
spectral_stepping = round(spectralStep / 1000 * samplefreq);

% Count the number of channels in an epoch
num_channels = EEG.nbchan;

% Create parameter arrays
parms = [modelOrd, hpCutoff + freqBinWidth / 2, ...
         lp_cutoff - freqBinWidth / 2, freqBinWidth, ...
         round(freqBinWidth / 0.2), trend, samplefreq];

memparms = parms;

if length(memparms) < 6
    memparms(6) = 0;
end

if length(memparms) < 7
    memparms(7) = 1;
end

memparms(8) = spectral_stepping;
memparms(9) = spectral_size / spectral_stepping;

start = parms(2);
stop = parms(3);
binwidth = parms(4);

spectral_bins = round((stop - start) / binwidth) + 1;

countall = 0;

countcond0 = 0;
countcond1 = 0;
countcond2 = 0;
countcond3 = 0;
countcond4 = 0; 

avgdata0 = zeros(spectral_bins, num_channels, countcond0);
avgdata1 = zeros(spectral_bins, num_channels, countcond1);
avgdata2 = zeros(spectral_bins, num_channels, countcond2);
avgdata3 = zeros(spectral_bins, num_channels, countcond3);
avgdata4 = zeros(spectral_bins, num_channels, countcond4);

% Calculate average spectra for T0
for i = 1:size(mergedEpochsT0.data, 3)
    condition0data = double(mergedEpochsT0.data(:, :, i)');
    [trialspectrum, freq_bins] = mem(condition0data, memparms);
    countall = countall + size(trialspectrum, 3);
    if size(trialspectrum, 3) > 0
        trialspectrum = mean(trialspectrum, 3);
        countcond0 = countcond0 + 1;
        avgdata0(:, :, countcond0) = trialspectrum;
    end
end


% ... Calculate average spectra for other conditions (T1 left, T1 both, T2 right, T2 both)
% Calculating average spectra for T1: left fist 
for i = 1 : size ( mergedEpochsT1_left.data , 3)
    condition1data = double(mergedEpochsT1_left.data(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition1data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond1 = countcond1+1;
        avgdata1(:, :, countcond1) = trialspectrum;
    end
    
end


% % Calculating average spectra for T1: both fists 
% for i = 1 : size ( mergedEpochsT1_both.data , 3)
%     condition2data = double(mergedEpochsT1_both.data(:,:,i)');
%     [trialspectrum, freq_bins] = mem(condition2data, memparms);
%     countall = countall + size(trialspectrum, 3 );
%     if( size( trialspectrum, 3 ) > 0 )
%         trialspectrum = mean( trialspectrum, 3 );
%         countcond2 = countcond2+1;
%         avgdata2(:, :, countcond2) = trialspectrum;
%     end
%     
% end


% Calculating average spectra for T2: right fists 
for i = 1 : size ( mergedEpochsT2_right.data, 3)
    condition3data = double(mergedEpochsT2_right.data(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition3data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond3 = countcond3+1;
        avgdata3(:, :, countcond3) = trialspectrum;
    end
    
end

% 
% % Calculating average spectra for T2: both feet 
% for i = 1 : size ( mergedEpochsT2_both.data , 3)
%     condition4data = double(mergedEpochsT2_both.data(:,:,i)');
%     [trialspectrum, freq_bins] = mem(condition4data, memparms);
%     countall = countall + size(trialspectrum, 3 );
%     if( size( trialspectrum, 3 ) > 0 )
%         trialspectrum = mean( trialspectrum, 3 );
%         countcond4 = countcond4+1;
%         avgdata4(:, :, countcond4) = trialspectrum;
%     end
% 
% end

%% Calculate r-squared and average amplitudes

% Calculate r-squared values for each condition and channel
res1 = mean(avgdata0, 3);
res2 = [];

ressq = calc_rsqu(double(avgdata0), double(avgdata1), 1);

res2 = mean(avgdata1, 3);

freq_bins = freq_bins - binwidth / 2;

%% Plot topographic maps

params.acqType = 'eeg';
params.topoParams = ['30'];
params.topoParams = eval(['[' params.topoParams ']']);

% %translate frequencies into bins - handle issue of user requesting
%  %the right edge of the final bin
 topoParams = params.topoParams;
%  %deal with right edge frequncies
  topoParams(topoParams == lp_cutoff) = freqBinWidth;
%  %translate into bins
  topoParams = (topoParams+freqBinWidth/2)/freqBinWidth;
  topofrequencybins=ceil(topoParams);

titleData = 'ciao';
handles = struct('r2', [], 'chans', [], 'topos', []);
topogrid = [1 2]; % Define topographic grid layout

switch length(params.topoParams)
      case 1
        topogrid = [1 1];
      case 2
        topogrid = [1 2];
      case {3 4}
        topogrid = [2 2];
      case {5 6}
        topogrid = [2 3];
      case {7 8 9}
        topogrid = [3 3];
      otherwise
        error([funcName ':maxTopoFreqsExceeded'], 'Maximum number of topographic plots exceeded');
    end

plotTopos(params.acqType, topofrequencybins,'%0.2f ms (%0.2f ms requested)', titleData, handles, params, topogrid, ressq, res1, res2);




%% Cacl_rsqu function from bci2000 trunk 
function [ressq, amp1, amp2] = calc_rsqu(data1, data2, rorrsqu)
%RSQU   [ressq, avgamp1, avgamp2]  = calc_rsqu(data1, data2, rorrsqu) calculates the r2-value for
%       two three-dimensional variables (dim1 by dim2 by trial) data1 and data2
%       the result is ressq (dim1, dim2); each element represents the r2 
%       for this particular combination of dim1, dim2 across trials
%       in addition to r2 values, this function also calculates average amplitudes 
%       for each sample and channel, for both data variables (i.e., conditions), and
%       returns these in amp1 and amp2
%       rorrsqu == 1 ... rsqu values
%                  2 ... r values

if (rorrsqu == 1)
   for ch=1:size(data1, 2)
    for samp=1:size(data1, 1)
       ressq(samp, ch)=rsqu(data1(samp, ch, :), data2(samp, ch, :));
       amp1(samp, ch)=mean(data1(samp, ch, :));
       amp2(samp, ch)=mean(data2(samp, ch, :));
    end
   end
else
   for ch=1:size(data1, 2)
    for samp=1:size(data1, 1)
       ressq(samp, ch)=rvalue(data1(samp, ch, :), data2(samp, ch, :));
       amp1(samp, ch)=mean(data1(samp, ch, :));
       amp2(samp, ch)=mean(data2(samp, ch, :));
    end
   end
end
end


%% RSQU function from BCI2000 trunk. Called in calc_rsqu

function erg = rsqu(q, r)
%RSQU   erg=rsqu(r, q) computes the r2-value for
%       two one-dimensional distributions given by
%       the vectors q and r


q=double(q);
r=double(r);

sum1=sum(q);
sum2=sum(r);
n1=length(q);
n2=length(r);
sumsqu1=sum(q.*q);
sumsqu2=sum(r.*r);

G=((sum1+sum2)^2)/(n1+n2);
den = sumsqu1+sumsqu2-G;
if( abs(den) < eps )
  erg = 0;
else
  erg=(sum1^2/n1+sum2^2/n2-G)/(sumsqu1+sumsqu2-G);
end

end


%% Plotting Topos function
function plotTopos(acqType, topoBins, pltTitle, titleData,handles, params, topogrid, ressq, res1, res2)
      if isempty(handles.topos)
        handles.topos = figure;
      else
        figure(handles.topos);
      end
      clf;
      set(handles.topos, 'name', 'Topographies');
      
      num_topos=length(params.topoParams);
      topoHandles = zeros(num_topos, 1);
      
      params.targetConditions{1}= 'left Swing';
      params.targetConditions{2}=' right swing';
      
      for cur_topo=1:num_topos
        pltIdx = cur_topo;
        hPlt = subplot(topogrid(1), topogrid(2), pltIdx);
        topoHandles(cur_topo) = hPlt;
        if length(params.targetConditions) == 2
          data2plot=ressq(topoBins(cur_topo), :);
        else
          data2plot = res1(topoBins(cur_topo), :);
        end
        
        options = {'maplimits', [min(min(data2plot)), max(max(data2plot))]};
        options = { 'gridscale', 200 };
        if strcmpi( acqType, 'eeg' )
          options = { options{:}, 'electrodes', 'labels' };
        end
        
        eloc_file = '/Users/vaastav/Desktop/ncan/motor imagery/data/files/eeg64.loc';
        
        
        topoplot(data2plot, eloc_file, options{:} );
        %titletxt=sprintf(pltTitle, titleData{cur_topo, :});
        %title(titletxt); 
        
        %colormap jet;
        
        if cur_topo == 1
          topoPos = get(hPlt, 'position');
        end
        
        if(cur_topo == num_topos)
          hCb = colorbar;
          
          %compensate for matlab deforming the graph showing the colorbar
          topoPosLast = get(hPlt, 'position');
          topoPosLast(3) = topoPos(3);
          set(gca, 'position', topoPosLast);
        end      
      end
      lastRow = 0;
      
      cbPos = get(hCb, 'position');
      bufLen = cbPos(1) - topoPosLast(1);
      
      for cur_topo=1:num_topos
        pltIdx = cur_topo;
        hPlt = topoHandles(cur_topo);
        pos = get(hPlt, 'position');
        
        if cur_topo==1
          shiftAmt = pos(1)/2;
        end
        if mod(pltIdx-1, topogrid(2)) == 0
          rowY = pos(2);
        else
          pos(2) == rowY;
        end
        
        pos(1) = pos(1)-shiftAmt;
        set(hPlt, 'position', pos);
      end
    
    end
