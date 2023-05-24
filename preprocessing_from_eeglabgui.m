

% Loop through the participants
for participant = 1:109
    % Loop through the runs for each participant
    for run = 1:14
        % Construct the filename for each run
        filename = sprintf('/Users/vaastav/Desktop/ncan/motor imagery/data/files/S%03d/S%03dR%02d.edf', participant, participant, run);

        % Load the dataset
        EEG = import_edf(filename, 0);
        
        % Preprocessing steps
        EEG = pop_reref(EEG, []);
        
        EEG = pop_rmbase(EEG, [], []);
        update_name = sprintf('S%03d/S%03dR%02d_basline_epochs_removed', participant, participant, run);
        EEG.setname = update_name;
        
        EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5);
        update_name = sprintf('S%03d/S%03dR%02d_basline_epochs_removed_0.5hp', participant, participant, run);
        EEG.setname = update_name;
      
        
        EEG = pop_cleanline(EEG, 'bandwidth', 2, 'chanlist', [1:64], 'computepower', 1, 'linefreqs', 60, 'newversion', 0, 'normSpectrum', 0, 'p', 0.01, 'pad', 2, 'plotfigures', 0, 'scanforlines', 0, 'sigtype', 'Channels', 'taperbandwidth', 2, 'tau', 100, 'verb', 1, 'winsize', 4, 'winstep', 1);
        
        EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, 'LineNoiseCriterion', 4, 'Highpass', 'off', 'BurstCriterion', 20, 'WindowCriterion', 0.25, 'BurstRejection', 'on', 'Distance', 'Euclidian', 'WindowCriterionTolerances', [-Inf 7]);
        
        % Save the preprocessed dataset
        outputFilename = sprintf('preprocessed_data_participant%d_run%d.set', participant, run);
        outputFilePath = '/Users/vaastav/Desktop/ncan/motor imagery/data/files/preprocessed';
        EEG = pop_saveset(EEG, 'filename', outputFilename, 'filepath', outputFilePath);
    end
end

