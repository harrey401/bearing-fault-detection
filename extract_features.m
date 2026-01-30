function extract_features_EARLY_DETECTION()
% EXTRACT FEATURES - Three-Stage Early Detection Labeling
% Labels: Healthy (0-60%), Degrading (60-90%), Critical (90-100%)

clearvars; close all; clc;

fprintf('\n=== FEATURE EXTRACTION - EARLY DETECTION ===\n\n');

%% Configuration
cfg.rawDir = 'data/raw';
cfg.outDir = 'data/processed';
cfg.outFile = 'features_early_detection.csv';
cfg.fs = 20000;

cfg.healthyClass = 'Healthy';
cfg.degradingClass = 'Degrading';
cfg.criticalClass = 'Critical';

% Three-stage labeling thresholds
cfg.healthyEnd = 0.60;    % 0-60%: Healthy
cfg.degradingEnd = 0.90;  % 60-90%: Degrading  
cfg.criticalEnd = 1.00;   % 90-100%: Critical

%% Gather Files
fprintf('[1/3] Gathering data files...\n');

runs = {'1st_test', '2nd_test', '4th_test'};
allFiles = {};
runLabels = {};

for r = 1:length(runs)
    runDir = fullfile(cfg.rawDir, runs{r});
    if ~isfolder(runDir)
        fprintf('  Warning: %s not found, skipping\n', runs{r});
        continue;
    end
    
    files = dir(fullfile(runDir, '*.txt'));
    if isempty(files)
        files = dir(fullfile(runDir, '2*.???.???.??'));
    end
    
    fprintf('  Found %d files in %s\n', length(files), runs{r});
    
    for f = 1:length(files)
        allFiles{end+1} = fullfile(runDir, files(f).name);
        runLabels{end+1} = runs{r};
    end
end

fprintf('  Total files: %d\n', length(allFiles));

%% Assign Labels
fprintf('\n[2/3] Assigning three-stage labels...\n');

labels = cell(length(allFiles), 1);
fileIndices = zeros(length(allFiles), 1);

for r = 1:length(runs)
    runMask = strcmp(runLabels, runs{r});
    runFiles = find(runMask);
    nFiles = length(runFiles);
    
    healthyEnd = round(cfg.healthyEnd * nFiles);
    degradingEnd = round(cfg.degradingEnd * nFiles);
    
    for i = 1:nFiles
        fileIdx = runFiles(i);
        fileIndices(fileIdx) = i;
        
        if i <= healthyEnd
            labels{fileIdx} = cfg.healthyClass;
        elseif i <= degradingEnd
            labels{fileIdx} = cfg.degradingClass;
        else
            labels{fileIdx} = cfg.criticalClass;
        end
    end
    
    fprintf('  %s: H(1-%d) D(%d-%d) C(%d-%d)\n', runs{r}, ...
            healthyEnd, healthyEnd+1, degradingEnd, degradingEnd+1, nFiles);
end

%% Extract Features
fprintf('\n[3/3] Extracting features...\n');

nFiles = length(allFiles);
nFeatures = 9;
X = zeros(nFiles, nFeatures);
fileNames = cell(nFiles, 1);
runIdx = zeros(nFiles, 1);

fprintf('  Progress: ');
lastPercent = 0;

for i = 1:nFiles
    try
        data = readmatrix(allFiles{i}, 'FileType', 'text');
        if isempty(data)
            error('Empty file');
        end
        
        signal = data(:, 1);
        X(i, :) = compute_features(signal, cfg.fs);
        fileNames{i} = allFiles{i};
        runIdx(i) = find(strcmp(runs, runLabels{i}), 1);
        
    catch ME
        fprintf('\n  Warning: Failed to process file %d: %s\n', i, ME.message);
        X(i, :) = NaN;
    end
    
    currentPercent = floor(100 * i / nFiles);
    if currentPercent >= lastPercent + 10
        fprintf('%d%% ', currentPercent);
        lastPercent = currentPercent;
    end
end

fprintf('100%%\n');
fprintf('  Feature extraction complete\n');

%% Save Results
if ~exist(cfg.outDir, 'dir')
    mkdir(cfg.outDir);
end

fprintf('\n[*] Creating features table...\n');

featNames = {'RMS', 'Kurtosis', 'Skewness', 'CrestFactor', 'PeakToPeak', ...
             'SpectralFlatness', 'BandEnergy1', 'BandEnergy2', 'BandEnergy3'};

T = array2table(X, 'VariableNames', featNames);
T.Label = categorical(labels);
T.Run = runIdx;
T.FileName = fileNames;
T.FileIndex = fileIndices;

outPath = fullfile(cfg.outDir, cfg.outFile);
writetable(T, outPath);

fprintf('  Saved to: %s\n', outPath);
fprintf('  Total samples: %d\n', height(T));

% Show final distribution
fprintf('\n  Final Class Distribution:\n');
for cls = categories(T.Label)'
    n = sum(T.Label == cls{1});
    fprintf('    %s: %d (%.1f%%)\n', cls{1}, n, 100*n/height(T));
end

%% Visualization
fprintf('\n[*] Creating visualization...\n');

figure('Position', [100 100 1400 900]);

for r = 1:length(runs)
    runMask = T.Run == r;
    if ~any(runMask)
        continue;
    end
    
    subplot(3, 1, r);
    
    runData = T(runMask, :);
    rms_vals = runData.RMS;
    file_idx = runData.FileIndex;
    
    % Plot with colors
    hold on;
    for j = 1:height(runData)
        if runData.Label(j) == cfg.healthyClass
            plot(file_idx(j), rms_vals(j), 'g.', 'MarkerSize', 8);
        elseif runData.Label(j) == cfg.degradingClass
            plot(file_idx(j), rms_vals(j), '.', 'Color', [1 0.6 0], 'MarkerSize', 8);
        else
            plot(file_idx(j), rms_vals(j), 'r.', 'MarkerSize', 8);
        end
    end
    
    % Add boundary lines
    nFiles_run = max(file_idx);
    healthyEnd = round(cfg.healthyEnd * nFiles_run);
    degradingEnd = round(cfg.degradingEnd * nFiles_run);
    
    yl = ylim;
    line([healthyEnd healthyEnd], yl, 'Color', 'g', 'LineStyle', '--', 'LineWidth', 2);
    line([degradingEnd degradingEnd], yl, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
    
    text(healthyEnd, yl(2)*0.95, ' Degradation Starts', ...
         'Color', 'g', 'FontSize', 10, 'FontWeight', 'bold');
    text(degradingEnd, yl(2)*0.95, ' Critical', ...
         'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
    
    xlabel('File Index');
    ylabel('RMS');
    title(sprintf('%s - Three-Stage Early Detection Labeling', runs{r}));
    legend('Healthy', 'Degrading (Early Warning)', 'Critical', 'Location', 'best');
    grid on;
end

figPath = fullfile(cfg.outDir, 'labeling_visualization.png');
saveas(gcf, figPath);
fprintf('  Visualization saved: %s\n', figPath);

fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');

end

%% Feature Computation
function F = compute_features(signal, fs)
% Compute 9 features from vibration signal

signal = signal(:);
signal = signal - mean(signal);

% Time-domain features
rms_val = sqrt(mean(signal.^2));
kurt_val = kurtosis(signal);
skew_val = skewness(signal);
peak_val = max(abs(signal));
p2p_val = max(signal) - min(signal);
crest_val = peak_val / (rms_val + eps);

% Frequency-domain features
N = numel(signal);
X = abs(fft(signal));
X = X(1:floor(N/2));

spec_flat = geomean(X + eps) / (mean(X) + eps);

f = (0:floor(N/2)-1)' * (fs / N);
band_energy1 = sum(X(f >= 500 & f < 1500).^2);
band_energy2 = sum(X(f >= 1500 & f < 3000).^2);
band_energy3 = sum(X(f >= 3000 & f < 6000).^2);

F = [rms_val, kurt_val, skew_val, crest_val, p2p_val, ...
     spec_flat, band_energy1, band_energy2, band_energy3];
end
