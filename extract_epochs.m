
% Loop through the participants
for participant = 1:1
    % Loop through the runs for each participant
    for run = 3:14
        % Construct the filename for each run
        filename = sprintf('/Users/vaastav/Desktop/ncan/motor imagery/data/files/preprocessed/preprocessed_data_participant%d_run%d.set', participant, run);

        % Load the preprocessed dataset
        EEG = pop_loadset('filename', filename);

        % Calculating epoch duration
        samplingRate = EEG.srate; % Obtain the sampling rate from the loaded dataset
        windowDurationInSeconds = 4.1; % Set the desired duration of each epoch window in seconds
        epochDuration = round(samplingRate * windowDurationInSeconds); % Calculate the epoch duration in data points
        
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
                case 'rest'
                    % Extract epoch for T0
                    epochEnd = min(EEG.event(j).latency + epochDuration - 1, size(EEG.data, 2));
                    epochs_T0 = [epochs_T0; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds])];
                case 'T1'
                    % Check the run number to determine if it's left fist or both fists
                    if ismember(run, [3, 4, 7, 8, 11, 12])
                        % Extract epoch for left fist (T1)
                        epochEnd = min(EEG.event(j).latency + epochDuration - 1, size(EEG.data, 2));
                        epochs_T1_left = [epochs_T1_left; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds])];
                    else
                        % Extract epoch for both fists (T1)
                        epochEnd = min(EEG.event(j).latency + epochDuration - 1, size(EEG.data, 2));
                        epochs_T1_both = [epochs_T1_both; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds])];
                    end
                case 'T2'
                    % Check the run number to determine if it's right fist or both feet
                    if ismember(run, [3, 4, 7, 8, 11, 12])
                        % Extract epoch for right fist (T2)
                        epochEnd = min(EEG.event(j).latency + epochDuration - 1, size(EEG.data, 2));
                        epochs_T2_right = [epochs_T2_right; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds])];
                    else
                        % Extract epoch for both feet (T2)
                        epochEnd = min(EEG.event(j).latency + epochDuration - 1, size(EEG.data, 2));
                        epochs_T2_both = [epochs_T2_both; pop_epoch(EEG, {eventType}, [0, windowDurationInSeconds])];
                    end
            end
        end
        
        % Save the epochs for each event separately
        save(sprintf('epochs_participant%d_run%d.mat', participant, run), 'epochs_T0', 'epochs_T1_left', 'epochs_T1_both', 'epochs_T2_right', 'epochs_T2_both');
    end 

%% --------------------------------------------------------------------------------



    %default:
spectralStep = 166;
spectralSize = 333;
 

% 1 modelOrd
modelOrd = 18+round(samplefreq/100);

% 2 settings.hpCutoff+settings.freqBinWidth/2
hpCutoff = -1;
freqBinWidth = 4;

%3 lp_cutoff-settings.freqBinWidth/2
lp_cutoff = 71;
maxFreqEEG = 71;

 if lp_cutoff > samplefreq/2
     lp_cutoff = samplefreq/2;
     %if the last bin has less samples, truncate it
     lp_cutoff = lp_cutoff - mod(lp_cutoff- hpCutoff, freqBinWidth);
 end

  %6 settings.trend
    trend = 1;

 %7 samplefreq
 samplefreq = EEG.srate;

 parms = [modelOrd, hpCutoff+ freqBinWidth/2, ...
     lp_cutoff-freqBinWidth/2, freqBinWidth, ... 
     round(freqBinWidth/.2), trend, samplefreq];

 %8 and 9
spectral_size = round(spectralSize/1000 * samplefreq);
spectral_stepping = round(spectralStep/1000 * samplefreq);

memparms(8) = spectral_stepping; %27%
memparms(9) = spectral_size/spectral_stepping; %(53/27)=1.9630

start=parms(2);
stop=parms(3);
binwidth=parms(4);

spectral_bins=round((stop-start)/binwidth)+1;

    avgdata0 = [];
    avgdata1 = [];
    avgdata2 = [];
    avgdata3 = [];
    avgdata4 = [];
    countall = 0;
    
    countcond0 = 0;
    countcond1 = 0;
    countcond2 = 0;
    countcond3 = 0;
    countcond4 = 0;
   
    % Count the number of channels in an epoch
    num_channels = EEG.nbchan;

% pre-allocate matrices to speed up computation
avgdata0=zeros(spectral_bins, num_channels, countcond0);
avgdata1=zeros(spectral_bins, num_channels, countcond1);
avgdata2=zeros(spectral_bins, num_channels, countcond2);
avgdata3=zeros(spectral_bins, num_channels, countcond3);
avgdata4=zeros(spectral_bins, num_channels, countcond4);

% Calculating average spectra for T0 
for i = 1 : size ( epochs_T0, 3)
    condition0data = double(epochs_T0(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition0data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond0 = countcond0+1;
        avgdata0(:, :, countcond0) = trialspectrum;
    end
    
end

% Calculating average spectra for T1: left fist 
for i = 1 : size ( epochs_T1_left , 3)
    condition1data = double(epochs_T1_left(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition1data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond1 = countcond1+1;
        avgdata1(:, :, countcond1) = trialspectrum;
    end
    
end


% Calculating average spectra for T1: both fists 
for i = 1 : size ( epochs_T1_both , 3)
    condition2data = double(epochs_T1_both(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition2data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond2 = countcond2+1;
        avgdata2(:, :, countcond2) = trialspectrum;
    end
    
end

% Calculating average spectra for T2: right fists 
for i = 1 : size ( epochs_T2_rigth, 3)
    condition3data = double(epochs_T2_right(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition3data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond3 = countcond3+1;
        avgdata3(:, :, countcond3) = trialspectrum;
    end
    
end

% Calculating average spectra for T2: both feet 
for i = 1 : size ( epochs_T2_both , 3)
    condition4data = double(epochs_T2_both(:,:,i)');
    [trialspectrum, freq_bins] = mem(condition4data, memparms);
    countall = countall + size(trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond4 = countcond4+1;
        avgdata4(:, :, countcond4) = trialspectrum;
    end
    
end

end 




%{ 



load('D:\0-MATLAB\1_Data\0_miNCAN_Data\miNCAN_results\SettingFile.mat','settings')

%load('D:\0-MATLAB\1_Data\0_miNCAN_Data\miNCAN_results\working.mat')

samplefreq = EEG.srate;

% 1 modelOrd
modelOrd = 18+round(samplefreq/100);

% 2 settings.hpCutoff+settings.freqBinWidth/2
settings.hpCutoff = -1;
settings.freqBinWidth = 4;


%3 lp_cutoff-settings.freqBinWidth/2

params.acqType = 'eeg';
settings.maxFreqEEG = 71;
settings.maxFreqECoG = 201;

 if strcmp(params.acqType, 'eeg')
     lp_cutoff = settings.maxFreqEEG;
 else
     lp_cutoff = settings.maxFreqECoG;
 end
 
 if lp_cutoff > samplefreq/2
     lp_cutoff = samplefreq/2;
     %if the last bin has less samples, truncate it
     lp_cutoff = lp_cutoff - mod(lp_cutoff-settings.hpCutoff, settings.freqBinWidth);
 end

 %4  settings.freqBinWidth
 
 %5 round(settings.freqBinWidth/.2)

 %6 settings.trend
 settings.trend = 1;

 %7 samplefreq
 

parms = [modelOrd, settings.hpCutoff+settings.freqBinWidth/2, ...
     lp_cutoff-settings.freqBinWidth/2, settings.freqBinWidth, ...
     round(settings.freqBinWidth/.2), settings.trend, samplefreq];

memparms = parms;
% 6 and 7 
if( length(memparms) < 6 )
  memparms(6) = 0;
end
if( length(memparms) < 7 )
  memparms(7) = 1;
end

%8 and 9
spectral_size = round(spectralSize/1000 * samplefreq);
spectral_stepping = round(spectralStep/1000 * samplefreq);

memparms(8) = spectral_stepping; %27%
memparms(9) = spectral_size/spectral_stepping; %(53/27)=1.9630

start=parms(2);
stop=parms(3);
binwidth=parms(4);

spectral_bins=round((stop-start)/binwidth)+1;
 
%%

%condition1data=double(signal(condition1idx, :));
avgdata1=[];
avgdata2=[];
countall=0;
countcond1=0;
countcond2=0;

% Count the number of channels in an epoch
num_channels=EEG.nbchan;


% pre-allocate matrices to speed up computation
avgdata1=zeros(spectral_bins, num_channels, countcond1);
avgdata2=zeros(spectral_bins, num_channels, countcond2);


for i = 1 : size ( EEGLS.data , 3)
    condition1data = double(EEGLS.data(:,:,i)');
    [trialspectrum, freq_bins] = mem( condition1data, memparms );
    countall = countall + size( trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond1=countcond1+1;
        avgdata1(:, :, countcond1)=trialspectrum;
    end
    
end


for i = 1 : size ( EEGRS.data , 3)
    condition2data = double(EEGRS.data(:,:,i)');
    [trialspectrum, freq_bins] = mem( condition2data, memparms );
    countall = countall + size( trialspectrum, 3 );
    if( size( trialspectrum, 3 ) > 0 )
        trialspectrum = mean( trialspectrum, 3 );
        countcond2=countcond2+1;
        avgdata2(:, :, countcond2)=trialspectrum;
    end
    
end

%%

% trials=unique(trialnr);
% trials = trials( trials > 0  );
% start=parms(2);
% stop=parms(3);
% binwidth=parms(4);
% num_channels=size(signal, 2);

spectral_bins=round((stop-start)/binwidth)+1;

%default:
spectralStep = 166;
spectralSize = 333;

%here changes according to sample freq = 160


start=parms(2);
stop=parms(3);
binwidth=parms(4);

memparms(8) = spectral_stepping; %27%
memparms(9) = spectral_size/spectral_stepping; %(53/27)=1.9630



%samplefreq = 160

% condition1data is the data [samples x channels] of one trial

[trialspectrum, freq_bins] = mem( condition1data, memparms );

% % pre-allocate matrices to speed up computation
% avgdata1=zeros(spectral_bins, num_channels, countcond1);
% if length(analysisParams.targetConditions) == 2
%   avgdata2=zeros(spectral_bins, num_channels, countcond2);
% end

%%
% calculate average spectra for each condition and each channel
res1 = mean(avgdata1, 3);
res2 = [];
ressq = [];


  % calculate rvalue/rsqu for each channel and each spectral bin between the
  % two conditions
ressq = calc_rsqu(double(avgdata1), double(avgdata2), 1);

res2=mean(avgdata2, 3);

freq_bins = freq_bins - binwidth/2;

%%
params.acqType = 'eeg';

params.topoParams = ['30']; 

params.topoParams = eval(['[' params.topoParams ']']);

% %translate frequencies into bins - handle issue of user requesting
%  %the right edge of the final bin
 topoParams = params.topoParams;
%  %deal with right edge frequncies
  topoParams(topoParams == lp_cutoff) = lp_cutoff-settings.freqBinWidth;
%  %translate into bins
  topoParams = (topoParams+settings.freqBinWidth/2)/settings.freqBinWidth;
  topofrequencybins=ceil(topoParams);


%topofrequencybins = 8;


titleData = 'ciao';

handles = struct('r2', [], 'chans', [], 'topos', []);
%titleData = mat2cell(reshape(freq_bins(topofrequencybins) + settings.freqBinWidth/2, ...
%          length(topofrequencybins), 1), ...
%          ones(length(topofrequencybins), 1), 1);
%titleData(:, 2) = mat2cell(reshape(params.topoParams, ...
%          length(topofrequencybins), 1), ...
%          ones(length(topofrequencybins), 1), 1);


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

    
    
end

%} 
