
%% 1-D CNN - FFT feature extraction

clc; clear; close all;

%% Load extracted FFT-1D-CNN features
load('X_FFT_1DCNN_features_IMS_1st_test.mat')

AA = X_cnn_all_channels;
numChannels = numel(AA);

fprintf('Total channels found: %d\n', numChannels);

%% RUL settings
baselineRatio  = 0.10;
smoothWindow   = 51;
thresholdK     = 3;
sustainLength  = 30;
samplingInterval_minutes = 10;

RUL_results_CNN = cell(numChannels,1);

%% Channel-wise RSE-based RUL calculation
for ch = 1:numChannels

    fprintf('\n=====================================\n');
    fprintf('Processing FFT-1D-CNN Channel %d\n', ch);
    fprintf('=====================================\n');

    X = AA{ch};
    [numSamples, numFeatures] = size(X);

    X(~isfinite(X)) = 0;

    %% Healthy baseline
    numBaseline = max(5, round(baselineRatio * numSamples));

    X_base = X(1:numBaseline, :);

    mu_base = mean(X_base, 1);
    sigma_base = std(X_base, 0, 1);
    sigma_base(sigma_base == 0) = eps;

    %% RSE calculation
    RSE = zeros(numSamples,1);

    for i = 1:numSamples
        relative_error = (X(i,:) - mu_base) ./ sigma_base;
        RSE(i) = sqrt(mean(relative_error.^2));
    end

    %% Smooth RSE
    localSmoothWindow = smoothWindow;

    if localSmoothWindow >= numSamples
        localSmoothWindow = max(3, round(numSamples/20));
    end

    RSE_smooth = smoothdata(RSE, 'movmean', localSmoothWindow);

    %% Threshold
    baseline_RSE = RSE_smooth(1:numBaseline);
    threshold = mean(baseline_RSE) + thresholdK * std(baseline_RSE);

    %% Degradation detection
    aboveThreshold = RSE_smooth > threshold;

    degradationPoint = NaN;

    for i = 1:(numSamples - sustainLength + 1)
        if all(aboveThreshold(i:i+sustainLength-1))
            degradationPoint = i;
            break;
        end
    end

    if isnan(degradationPoint)
        warning('Channel %d did not cross threshold.', ch);
        degradationPoint = numSamples;
    end

    %% RUL calculation
    EOL = numSamples;

    RUL_samples = zeros(numSamples,1);

    for i = 1:numSamples
        RUL_samples(i) = EOL - i;
    end

    RUL_samples(RUL_samples < 0) = 0;

    RUL_minutes = RUL_samples * samplingInterval_minutes;
    RUL_hours   = RUL_minutes / 60;
    RUL_days    = RUL_hours / 24;

    degradationTime_minutes = degradationPoint * samplingInterval_minutes;
    degradationTime_hours   = degradationTime_minutes / 60;
    degradationTime_days    = degradationTime_hours / 24;

    totalLife_minutes = EOL * samplingInterval_minutes;
    totalLife_hours   = totalLife_minutes / 60;
    totalLife_days    = totalLife_hours / 24;

    remainingAfterDegradation_samples = EOL - degradationPoint;
    remainingAfterDegradation_minutes = remainingAfterDegradation_samples * samplingInterval_minutes;
    remainingAfterDegradation_hours   = remainingAfterDegradation_minutes / 60;
    remainingAfterDegradation_days    = remainingAfterDegradation_hours / 24;

    %% Store results
    RUL_results_CNN{ch}.RSE = RSE;
    RUL_results_CNN{ch}.RSE_smooth = RSE_smooth;
    RUL_results_CNN{ch}.threshold = threshold;
    RUL_results_CNN{ch}.degradationPoint = degradationPoint;

    RUL_results_CNN{ch}.RUL_samples = RUL_samples;
    RUL_results_CNN{ch}.RUL_hours = RUL_hours;
    RUL_results_CNN{ch}.RUL_days = RUL_days;

    RUL_results_CNN{ch}.degradationTime_hours = degradationTime_hours;
    RUL_results_CNN{ch}.degradationTime_days = degradationTime_days;

    RUL_results_CNN{ch}.remainingAfterDegradation_hours = remainingAfterDegradation_hours;
    RUL_results_CNN{ch}.remainingAfterDegradation_days = remainingAfterDegradation_days;

    RUL_results_CNN{ch}.totalLife_hours = totalLife_hours;
    RUL_results_CNN{ch}.totalLife_days = totalLife_days;

    RUL_results_CNN{ch}.numSamples = numSamples;
    RUL_results_CNN{ch}.numFeatures = numFeatures;
    RUL_results_CNN{ch}.EOL = EOL;
end

%% Dynamic labels
channelLabels = cell(numChannels,1);

for ch = 1:numChannels
    channelLabels{ch} = ['Ch ', num2str(ch)];
end

%% Figure 1: Combined RSE trend
figure(1);
clf;
hold on;

for ch = 1:numChannels
    plot(RUL_results_CNN{ch}.RSE_smooth, 'LineWidth', 1.8);
end

xlabel('Run Index');
ylabel('Smoothed RSE');
title('Combined Channel-Wise RSE Degradation Trend Using FFT-1D-CNN Features');
legend(channelLabels, 'Location', 'best');
grid on;
hold off;

saveas(gcf, 'Figure_1_Combined_RSE_FFT_1DCNN.jpg');

%% Build summary arrays
degPoint = zeros(numChannels,1);
degHours = zeros(numChannels,1);
degDays  = zeros(numChannels,1);
remainingHours = zeros(numChannels,1);
remainingDays  = zeros(numChannels,1);

for ch = 1:numChannels

    degPoint(ch) = RUL_results_CNN{ch}.degradationPoint;
    degHours(ch) = RUL_results_CNN{ch}.degradationTime_hours;
    degDays(ch)  = RUL_results_CNN{ch}.degradationTime_days;

    remainingHours(ch) = RUL_results_CNN{ch}.remainingAfterDegradation_hours;
    remainingDays(ch)  = RUL_results_CNN{ch}.remainingAfterDegradation_days;
end

%% Print summary table
fprintf('\n=========================================================\n');
fprintf('RUL TIME-FRAME SUMMARY FOR IMS 1ST TEST USING FFT-1D-CNN\n');
fprintf('=========================================================\n');
fprintf('Channel\tDeg.Point\tDeg.Hour\tDeg.Day\tRemaining.Hour\tRemaining.Day\n');

for ch = 1:numChannels
    fprintf('%d\t%d\t\t%.2f\t\t%.2f\t%.2f\t\t%.2f\n', ...
        ch, ...
        degPoint(ch), ...
        degHours(ch), ...
        degDays(ch), ...
        remainingHours(ch), ...
        remainingDays(ch));
end

%% Create summary table
SummaryTable_CNN = table((1:numChannels)', ...
    degPoint, ...
    degHours, ...
    degDays, ...
    remainingHours, ...
    remainingDays, ...
    'VariableNames', {'Channel', ...
    'DegradationPoint', ...
    'DegradationHour', ...
    'DegradationDay', ...
    'RemainingHour', ...
    'RemainingDay'});

disp(SummaryTable_CNN);

writetable(SummaryTable_CNN, ...
    'RUL_TimeFrame_Summary_FFT_1DCNN_IMS_1st_Test.csv');

%% Figure 2: Box plot visualization
rng(10);

boxData = [];
groupData = [];

spreadRatio = 0.30;
numRep = 40;

for ch = 1:numChannels

    centerValue = degHours(ch);
    spreadValue = max(5, spreadRatio * centerValue);

    tempData = centerValue + spreadValue * randn(numRep,1);
    tempData(tempData < 0) = 0.10 * centerValue;

    boxData = [boxData; tempData];
    groupData = [groupData; ch * ones(numRep,1)];
end

figure(2);
clf;
set(gcf, 'Position', [100 100 1400 800]);

boxplot(boxData, groupData, ...
    'Labels', channelLabels, ...
    'Symbol', '', ...
    'Widths', 0.55);

hold on;

scatter(1:numChannels, degHours, ...
    90, 'filled', ...
    'MarkerEdgeColor', 'k');

for ch = 1:numChannels
    text(ch, degHours(ch) + 12, ...
        sprintf('%.2f h', degHours(ch)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 12, ...
        'FontWeight', 'bold');
end

xlabel('Channel', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Degradation Time (Hours)', 'FontSize', 16, 'FontWeight', 'bold');

title('Degradation Time Using FFT-1D-CNN Features - IMS 1st Test', ...
    'FontSize', 20, ...
    'FontWeight', 'bold');

grid on;
box on;

if max(degHours) > 0
    ylim([0, max(degHours)*1.35]);
end

set(gca, 'FontSize', 14, 'LineWidth', 1.2);

summaryText = sprintf('FFT-1D-CNN Degradation Time Summary\n');

for ch = 1:numChannels
    summaryText = sprintf('%sCh %d: %.2f h   (%.2f days)\n', ...
        summaryText, ch, degHours(ch), degDays(ch));
end

annotation('textbox', [0.73 0.57 0.22 0.32], ...
    'String', summaryText, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', 'white', ...
    'EdgeColor', [0.4 0.4 0.4], ...
    'FontSize', 12, ...
    'FontWeight', 'bold');

annotation('textbox', [0.22 0.01 0.60 0.04], ...
    'String', 'Box shows synthetic spread for visualization; markers/text show actual detected degradation time from FFT-1D-CNN features.', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12);

hold off;

saveas(gcf, 'Figure_2_BoxPlot_Degradation_Time_FFT_1DCNN.jpg');
savefig(gcf, 'Figure_2_BoxPlot_Degradation_Time_FFT_1DCNN.fig');

%% Figure 3: Actual degradation-time bar plot
figure(3);
clf;

bar(degHours);

xlabel('Channel');
ylabel('Degradation Time (Hours)');
title('Actual Detected Degradation Time by Channel Using FFT-1D-CNN Features');

xticks(1:numChannels);
xticklabels(channelLabels);

grid on;

for ch = 1:numChannels
    text(ch, degHours(ch) + 5, ...
        sprintf('%.2f h', degHours(ch)), ...
        'HorizontalAlignment','center', ...
        'FontSize',10, ...
        'FontWeight','bold');
end

saveas(gcf, 'Figure_3_Actual_Degradation_Time_FFT_1DCNN_Bar.jpg');

%% Save MATLAB results
save('RUL_results_FFT_1DCNN_IMS_1st_Test.mat', ...
    'RUL_results_CNN', ...
    'SummaryTable_CNN', ...
    'degPoint', ...
    'degHours', ...
    'degDays', ...
    'remainingHours', ...
    'remainingDays', ...
    'baselineRatio', ...
    'smoothWindow', ...
    'thresholdK', ...
    'sustainLength', ...
    'samplingInterval_minutes', ...
    '-v7.3');

fprintf('\nFFT-1D-CNN RUL analysis completed and saved successfully.\n');
