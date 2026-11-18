
clc; clear; close all;

%% ============================================================
%% FFT + Bayesian Optimized 1-D CNN Feature Extraction
%% IMS 3rd Test Dataset
%% No FFT normalization
%% ============================================================

%% Load dataset
load IMS_4th_test_Runs.mat   % variable: Runs

AllRuns = Runs;
clear Runs

%% Dataset information
numRuns = numel(AllRuns);
[numRows, numChannels] = size(AllRuns{1});

fprintf('Total runs/cells: %d\n', numRuns);
fprintf('Each run contains %d samples and %d channels.\n', ...
    numRows, numChannels);

%% ============================================================
%% Split dataset
%% ============================================================

numProject = round(0.10 * numRuns);

earlyIdx = 1:numProject;
lateIdx  = (numRuns-numProject+1):numRuns;

excludeIdx = [earlyIdx lateIdx];
featureIdx = setdiff(1:numRuns, excludeIdx);

X_feature = AllRuns(featureIdx);

fprintf('Early training runs: %d\n', numel(earlyIdx));
fprintf('Late training runs: %d\n', numel(lateIdx));
fprintf('Feature extraction runs: %d\n', numel(X_feature));

%% ============================================================
%% FFT parameters
%% ============================================================

Fs = 20480;
L  = numRows;

freq = Fs*(0:(L/2))/L;
numFFTPoints = length(freq);

fprintf('FFT feature length = %d\n', numFFTPoints);

%% ============================================================
%% Prepare FFT-CNN training data
%% ============================================================

fprintf('\nPreparing FFT-CNN training data...\n');

[XTrain, YTrain] = prepareFFTTrainingData( ...
    AllRuns, ...
    earlyIdx, ...
    lateIdx, ...
    numFFTPoints);

fprintf('Training data prepared.\n');
fprintf('XTrain size: %s\n', mat2str(size(XTrain)));
fprintf('YTrain size: %s\n', mat2str(size(YTrain)));

%% ============================================================
%% Bayesian Optimization
%% ============================================================

fprintf('\nStarting Bayesian optimization...\n');

optimVars = [
    optimizableVariable('InitialLearnRate',[1e-5 1e-2], ...
        'Transform','log')

    optimizableVariable('MiniBatchSize',[16 128], ...
        'Type','integer')

    optimizableVariable('L2Regularization',[1e-6 1e-2], ...
        'Transform','log')

    optimizableVariable('FeatureDim',[32 128], ...
        'Type','integer')

    optimizableVariable('NumFilters1',[8 32], ...
        'Type','integer')

    optimizableVariable('NumFilters2',[16 64], ...
        'Type','integer')

    optimizableVariable('NumFilters3',[32 128], ...
        'Type','integer')
];

ObjFcn = @(optVars) cnnObjectiveFunction( ...
    optVars, XTrain, YTrain, numFFTPoints);

bayesResults = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations',30, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel',false, ...
    'PlotFcn',{@plotObjectiveModel,@plotMinObjective});

bestVars = bestPoint(bayesResults);

fprintf('\nBest Bayesian Optimization Parameters:\n');
disp(bestVars);

%% ============================================================
%% Train Final Optimized FFT-CNN
%% ============================================================

featureDim = bestVars.FeatureDim;

layers = createFFTCNNLayers( ...
    numFFTPoints, ...
    bestVars.FeatureDim, ...
    bestVars.NumFilters1, ...
    bestVars.NumFilters2, ...
    bestVars.NumFilters3);

options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize',bestVars.MiniBatchSize, ...
    'InitialLearnRate',bestVars.InitialLearnRate, ...
    'L2Regularization',bestVars.L2Regularization, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

fprintf('\nTraining final Bayesian optimized FFT-CNN...\n');

net = trainNetwork(XTrain, YTrain, layers, options);

fprintf('Final optimized CNN training completed.\n');

%% ============================================================
%% Extract FFT-CNN Features
%% ============================================================

X_cnn_all_channels = cell(numChannels,1);

fprintf('\nExtracting FFT-CNN features...\n');

for ch = 1:numChannels

    fprintf('\n========================================\n');
    fprintf('Processing channel %d of %d\n', ch, numChannels);
    fprintf('========================================\n');

    channelFeatures = zeros(numel(X_feature), ...
        featureDim, ...
        'single');

    for i = 1:numel(X_feature)

        rawSignal = X_feature{i}(:,ch);

        P1 = computeFFT(rawSignal, L);

        XInput = reshape(P1, numFFTPoints, 1, 1, 1);

        feat = activations(net, ...
            XInput, ...
            'cnn_features', ...
            'OutputAs','rows');

        channelFeatures(i,:) = single(feat);

    end

    X_cnn_all_channels{ch} = channelFeatures;

    fprintf('Feature size: %d x %d\n', ...
        size(channelFeatures,1), ...
        size(channelFeatures,2));
end

%% ============================================================
%% Combine All Channel Features
%% ============================================================

X_cnn_combined = [];

for ch = 1:numChannels
    X_cnn_combined = [X_cnn_combined, X_cnn_all_channels{ch}];
end

fprintf('\n========================================\n');
fprintf('FFT-CNN feature extraction completed.\n');
fprintf('Combined feature size: %d x %d\n', ...
    size(X_cnn_combined,1), ...
    size(X_cnn_combined,2));
fprintf('========================================\n');

%% ============================================================
%% Save Outputs
%% ============================================================

save('X_FFT_1DCNN_features_IMS_3rd_test_BayesOpt_NoNorm.mat', ...
    'net', ...
    'bayesResults', ...
    'bestVars', ...
    'X_cnn_all_channels', ...
    'X_cnn_combined', ...
    'freq', ...
    'featureIdx', ...
    'numFFTPoints', ...
    'featureDim', ...
    'numProject', ...
    'numRuns', ...
    'numRows', ...
    'numChannels', ...
    '-v7.3');

fprintf('\nResults saved successfully.\n');