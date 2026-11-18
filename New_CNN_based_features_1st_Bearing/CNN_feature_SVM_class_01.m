%% IMS 1st Test Bearing Fault Diagnosis
%% Gaussian/Bayesian Optimized Multiclass SVM
%% Using Bayesian Optimized FFT-1D-CNN Features
%% No FFT Normalization Version

clc; clear; close all;

%% ============================================================
%% Load Bayesian Optimized FFT-1D-CNN Features
%% ============================================================

load('X_FFT_1DCNN_features_IMS_1st_test_BayesOpt_NoNorm.mat')

AA = X_cnn_all_channels;
numChannels = numel(AA);

fprintf('Total channels found: %d\n', numChannels);
fprintf('CNN feature dimension: %d\n', featureDim);

%% ============================================================
%% Class Definitions
%% Ch 1-4 = Healthy
%% Ch 5-6 = Bearing 3 Inner Race Fault
%% Ch 7-8 = Bearing 4 Roller Element Fault
%% ============================================================

classNames = {'Healthy', 'InnerRaceFault', 'RollerElementFault'};
numClasses = numel(classNames);

channelClass = [ ...
    1; 1; 1; 1; ...
    2; 2; ...
    3; 3];

%% ============================================================
%% Equalize Feature Dimensions
%% ============================================================

featureDims = zeros(numChannels,1);

for ch = 1:numChannels
    featureDims(ch) = size(AA{ch},2);
end

minFeatureDim = min(featureDims);

fprintf('Minimum common feature dimension: %d\n', minFeatureDim);

%% ============================================================
%% Build Full Dataset
%% ============================================================

X_all = [];
Y_all = [];
channelID = [];

for ch = 1:numChannels

    Xch = AA{ch};

    Xch(~isfinite(Xch)) = 0;

    Xch = Xch(:,1:minFeatureDim);

    numSamples = size(Xch,1);

    X_all = [X_all; Xch];

    Y_all = [Y_all; repmat(channelClass(ch), numSamples, 1)];

    channelID = [channelID; repmat(ch, numSamples, 1)];
end

Y_cat = categorical(Y_all, 1:numClasses, classNames);

fprintf('Total samples: %d\n', size(X_all,1));
fprintf('Total features used: %d\n', size(X_all,2));

%% ============================================================
%% Multiple Shuffled Runs
%% ============================================================

numSVMRuns = 5;

allAccuracy  = zeros(numSVMRuns,1);
allPrecision = zeros(numSVMRuns,numClasses);
allRecall    = zeros(numSVMRuns,numClasses);
allF1        = zeros(numSVMRuns,numClasses);

AllRunResults = table;

%% ============================================================
%% Loop Over Runs
%% ============================================================

for run = 1:numSVMRuns

    fprintf('\n====================================================\n');
    fprintf('BayesOpt-NoNorm FFT-1D-CNN SVM RUN %d / %d\n', ...
        run, numSVMRuns);
    fprintf('====================================================\n');

    rng('shuffle');

    %% Train-test split
    cv = cvpartition(Y_cat, 'HoldOut', 0.30);

    X_train = X_all(training(cv), :);
    Y_train = Y_cat(training(cv));

    X_test = X_all(test(cv), :);
    Y_test = Y_cat(test(cv));

    %% Train Gaussian Bayesian Optimized Multiclass SVM
    template = templateSVM( ...
        'KernelFunction', 'gaussian', ...
        'Standardize', true);

    SVM_Model = fitcecoc( ...
        X_train, ...
        Y_train, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', categorical(classNames), ...
        'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', 30, ...
            'ShowPlots', false, ...
            'Verbose', 0, ...
            'Kfold', 5));

    %% Prediction
    Y_pred = predict(SVM_Model, X_test);

    %% Accuracy
    accuracy = mean(Y_pred == Y_test) * 100;

    allAccuracy(run) = accuracy;

    fprintf('Run %d Accuracy = %.2f %%\n', run, accuracy);

    %% Confusion Matrix
    figure;
    confusionchart(Y_test, Y_pred);

    title(sprintf(['BayesOpt-NoNorm FFT-1D-CNN + Gaussian SVM ', ...
        'Run %d Accuracy = %.2f%%'], run, accuracy));

    saveas(gcf, ...
        sprintf('BayesOpt_NoNorm_FFT_1DCNN_Run_%d_SVM_ConfusionMatrix.jpg', run));

    savefig(gcf, ...
        sprintf('BayesOpt_NoNorm_FFT_1DCNN_Run_%d_SVM_ConfusionMatrix.fig', run));

    close;

    %% Performance Metrics
    confMat = confusionmat(Y_test, Y_pred);

    precision = diag(confMat) ./ sum(confMat,1)';
    recall    = diag(confMat) ./ sum(confMat,2);
    F1        = 2 * (precision .* recall) ./ (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    F1(isnan(F1)) = 0;

    allPrecision(run,:) = precision';
    allRecall(run,:)    = recall';
    allF1(run,:)        = F1';

    %% Save Per-Run Classification Report
    ResultTable = table(classNames', precision, recall, F1, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});

    writetable(ResultTable, ...
        sprintf('BayesOpt_NoNorm_FFT_1DCNN_Run_%d_Classification_Report.csv', run));

    %% Save Per-Run Model
    save(sprintf('BayesOpt_NoNorm_FFT_1DCNN_Run_%d_SVM_Model.mat', run), ...
        'SVM_Model', ...
        'accuracy', ...
        'ResultTable', ...
        'confMat');

    %% Save Run Summary
    tempTable = table(run, accuracy, ...
        'VariableNames', {'Run', 'Accuracy'});

    AllRunResults = [AllRunResults; tempTable];

end

%% ============================================================
%% Final Statistics
%% ============================================================

meanAccuracy = mean(allAccuracy);
stdAccuracy  = std(allAccuracy);

fprintf('\n====================================================\n');
fprintf('FINAL BayesOpt-NoNorm FFT-1D-CNN + SVM RESULTS\n');
fprintf('====================================================\n');
fprintf('Mean Accuracy = %.2f %%\n', meanAccuracy);
fprintf('Std Accuracy  = %.2f %%\n', stdAccuracy);

%% Mean Precision, Recall, F1
meanPrecision = mean(allPrecision,1);
meanRecall    = mean(allRecall,1);
meanF1        = mean(allF1,1);

FinalMetricTable = table( ...
    classNames', ...
    meanPrecision', ...
    meanRecall', ...
    meanF1', ...
    'VariableNames', ...
    {'Class', 'MeanPrecision', 'MeanRecall', 'MeanF1'});

disp(FinalMetricTable);

%% ============================================================
%% Save Overall Results
%% ============================================================

writetable(AllRunResults, ...
    'BayesOpt_NoNorm_FFT_1DCNN_All_Run_Accuracy_Results.csv');

writetable(FinalMetricTable, ...
    'BayesOpt_NoNorm_FFT_1DCNN_Final_Mean_Performance_Metrics.csv');

%% ============================================================
%% Accuracy Plot
%% ============================================================

figure;
plot(1:numSVMRuns, allAccuracy, '-o', 'LineWidth', 2);

xlabel('Run Number');
ylabel('Accuracy (%)');

title(sprintf(['BayesOpt-NoNorm FFT-1D-CNN + Gaussian Optimized SVM ', ...
    'Accuracy Across %d Runs'], numSVMRuns));

grid on;

saveas(gcf, 'BayesOpt_NoNorm_FFT_1DCNN_All_Run_Accuracy_Plot.jpg');
savefig(gcf, 'BayesOpt_NoNorm_FFT_1DCNN_All_Run_Accuracy_Plot.fig');

%% ============================================================
%% Save Overall Workspace
%% ============================================================

save('BayesOpt_NoNorm_FFT_1DCNN_All_Run_SVM_Results.mat', ...
    'allAccuracy', ...
    'allPrecision', ...
    'allRecall', ...
    'allF1', ...
    'meanAccuracy', ...
    'stdAccuracy', ...
    'FinalMetricTable', ...
    'AllRunResults', ...
    'classNames', ...
    'channelClass', ...
    'channelID', ...
    'minFeatureDim', ...
    'featureDim', ...
    'bestVars', ...
    'bayesResults', ...
    '-v7.3');

fprintf('\nAll 5 BayesOpt-NoNorm FFT-1D-CNN + Gaussian SVM runs completed successfully.\n');