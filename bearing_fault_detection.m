function bearing_fault_detection_3class()
% BEARING FAULT DETECTION - Three-Class Early Warning System
% Detects: Healthy (0-60%), Degrading (60-90%), Critical (90-100%)

clearvars; close all; clc;

fprintf('\n=== THREE-CLASS BEARING FAULT DETECTION ===\n');
fprintf('Early Warning System for Predictive Maintenance\n\n');

%% Configuration
cfg.dataFile = 'data/processed/features_early_detection.csv';
cfg.outDir = 'results_3class';

%% Load Data
fprintf('Loading data...\n');
if ~isfile(cfg.dataFile)
    error('Features file not found! Run extract_features_EARLY_DETECTION() first.');
end

T = readtable(cfg.dataFile);
fprintf('  Loaded %d samples with %d features\n', height(T), width(T)-4);

% Ensure categorical labels
if ~iscategorical(T.Label)
    T.Label = categorical(T.Label);
end

% Display class distribution
fprintf('\n  Class Distribution:\n');
classNames = categories(T.Label);
for i = 1:length(classNames)
    n = sum(T.Label == classNames{i});
    fprintf('    %s: %d samples (%.1f%%)\n', classNames{i}, n, 100*n/height(T));
end

%% Split Data
fprintf('\n Splitting data (60%% train / 20%% val / 20%% test)...\n');

featCols = setdiff(T.Properties.VariableNames, {'Label', 'Run', 'FileName', 'FileIndex'});
X = T{:, featCols};
y = T.Label;

rng(42);
n = height(T);
idx = randperm(n);

nTrain = round(0.6 * n);
nVal = round(0.2 * n);

idxTrain = idx(1:nTrain);
idxVal = idx(nTrain+1:nTrain+nVal);
idxTest = idx(nTrain+nVal+1:end);

X_train = X(idxTrain, :);
y_train = y(idxTrain);
X_val = X(idxVal, :);
y_val = y(idxVal);
X_test = X(idxTest, :);
y_test = y(idxTest);

fprintf('  Training: %d samples\n', length(idxTrain));
fprintf('  Validation: %d samples\n', length(idxVal));
fprintf('  Test: %d samples\n', length(idxTest));

%% Normalize Features
fprintf('\n Normalizing features...\n');

mu_train = mean(X_train, 1, 'omitnan');
sigma_train = std(X_train, 0, 1, 'omitnan');
sigma_train(sigma_train == 0) = 1;

X_train_norm = (X_train - mu_train) ./ sigma_train;
X_val_norm = (X_val - mu_train) ./ sigma_train;
X_test_norm = (X_test - mu_train) ./ sigma_train;

fprintf('  Features normalized using training statistics\n');

%% Train Model
fprintf('\n Training multi-class SVM...\n');

% Get class names and counts
classNames = categories(y_train);
classCounts = zeros(length(classNames), 1);
for i = 1:length(classNames)
    classCounts(i) = sum(y_train == classNames{i});
end

% Calculate class weights (balance classes)
weights = zeros(length(y_train), 1);
maxCount = max(classCounts);
for i = 1:length(classNames)
    classWeight = maxCount / classCounts(i);
    weights(y_train == classNames{i}) = classWeight;
end

% Train SVM
template = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 1, ...
                       'KernelScale', 'auto', 'Standardize', false);

mdl = fitcecoc(X_train_norm, y_train, 'Learners', template, ...
               'Weights', weights, 'Coding', 'onevsall');

fprintf('  Training complete\n');

%% Evaluate on Validation Set
fprintf('\n Evaluating on validation set...\n');

y_val_pred = predict(mdl, X_val_norm);
C_val = confusionmat(y_val, y_val_pred, 'Order', classNames);
acc_val = sum(diag(C_val)) / sum(C_val(:));

fprintf('  Validation Accuracy: %.1f%%\n', 100*acc_val);

%% Final Evaluation on Test Set
fprintf('\n Final evaluation on test set...\n');

[y_test_pred, scores_test] = predict(mdl, X_test_norm);
C_test = confusionmat(y_test, y_test_pred, 'Order', classNames);
acc_test = sum(diag(C_test)) / sum(C_test(:));

fprintf('\n  TEST SET ACCURACY: %.1f%%\n\n', 100*acc_test);

% Per-class metrics
fprintf('  Per-Class Performance:\n');
for i = 1:length(classNames)
    tp = C_test(i, i);
    fp = sum(C_test(:, i)) - tp;
    fn = sum(C_test(i, :)) - tp;
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * precision * recall / (precision + recall);
    
    fprintf('    %s:\n', classNames{i});
    fprintf('      Precision: %.3f | Recall: %.3f | F1: %.3f\n', ...
            precision, recall, f1);
end

% Early detection analysis
degrading_idx = find(strcmp(classNames, 'Degrading'));
if ~isempty(degrading_idx)
    degrading_detected = C_test(degrading_idx, degrading_idx);
    degrading_total = sum(C_test(degrading_idx, :));
    degrading_recall = degrading_detected / degrading_total;
    
    fprintf('\n  EARLY DETECTION ANALYSIS:\n');
    fprintf('    Degrading Detection Rate: %.1f%%\n', 100*degrading_recall);
    fprintf('    Caught %d out of %d degrading bearings early\n', ...
            degrading_detected, degrading_total);
    
    % Missed detections
    healthy_idx = find(strcmp(classNames, 'Healthy'));
    critical_idx = find(strcmp(classNames, 'Critical'));
    
    if ~isempty(degrading_idx) && ~isempty(healthy_idx)
        deg_to_healthy = C_test(degrading_idx, healthy_idx);
        fprintf('    Degrading missed as Healthy: %d (%.1f%%)\n', ...
                deg_to_healthy, 100*deg_to_healthy/degrading_total);
    end
    
    if ~isempty(critical_idx) && ~isempty(healthy_idx)
        crit_to_healthy = C_test(critical_idx, healthy_idx);
        crit_total = sum(C_test(critical_idx, :));
        fprintf('    Critical missed as Healthy: %d (%.1f%%)\n', ...
                crit_to_healthy, 100*crit_to_healthy/crit_total);
    end
end

%% Save Results
if ~exist(cfg.outDir, 'dir')
    mkdir(cfg.outDir);
end

fprintf('\n[*] Saving results...\n');

% Save model
save(fullfile(cfg.outDir, 'model_3class.mat'), 'mdl', 'mu_train', 'sigma_train', ...
     'featCols', 'classNames', 'cfg');

% Save predictions
results = table(y_test, y_test_pred, scores_test, ...
                'VariableNames', {'Actual', 'Predicted', 'Scores'});
writetable(results, fullfile(cfg.outDir, 'test_predictions.csv'));

% Save metrics
metrics = struct();
metrics.test_accuracy = acc_test;
metrics.confusion_matrix = C_test;
metrics.class_names = classNames;
if exist('degrading_recall', 'var')
    metrics.degrading_recall = degrading_recall;
end
save(fullfile(cfg.outDir, 'metrics.mat'), 'metrics');

%% Generate Visualizations
fprintf('[*] Generating visualizations...\n');

% Confusion Matrix
figure('Position', [100 100 800 700]);
cm = confusionchart(C_test, classNames);
cm.Title = 'Confusion Matrix - Test Set';
cm.FontSize = 12;
saveas(gcf, fullfile(cfg.outDir, 'confusion_matrix.png'));

% Per-Class Performance
figure('Position', [100 100 1000 600]);
metrics_data = zeros(length(classNames), 3);
for i = 1:length(classNames)
    tp = C_test(i, i);
    fp = sum(C_test(:, i)) - tp;
    fn = sum(C_test(i, :)) - tp;
    
    metrics_data(i, 1) = tp / (tp + fp);  % Precision
    metrics_data(i, 2) = tp / (tp + fn);  % Recall
    metrics_data(i, 3) = 2 * metrics_data(i,1) * metrics_data(i,2) / ...
                         (metrics_data(i,1) + metrics_data(i,2));  % F1
end

bar(metrics_data);
set(gca, 'XTickLabel', classNames);
ylabel('Score');
ylim([0 1]);
legend('Precision', 'Recall', 'F1-Score', 'Location', 'best');
h = title('Per-Class Performance Metrics');
set(h, 'FontSize', 14, 'FontWeight', 'bold');
grid on;
saveas(gcf, fullfile(cfg.outDir, 'performance_metrics.png'));

fprintf('  Saved to %s/\n', cfg.outDir);

%% Summary
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('  Test Accuracy: %.1f%%\n', 100*acc_test);
if exist('degrading_recall', 'var')
    fprintf('  Early Detection Rate: %.1f%%\n', 100*degrading_recall);
end
fprintf('  Model saved: %s/model_3class.mat\n', cfg.outDir);
fprintf('  Figures saved: %s/*.png\n', cfg.outDir);

end
