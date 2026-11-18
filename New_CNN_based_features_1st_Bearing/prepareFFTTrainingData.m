%% ============================================================
%% Local Function 1: Prepare FFT Training Data
%% ============================================================

function [XTrain, YTrain] = prepareFFTTrainingData( ...
    AllRuns, earlyIdx, lateIdx, numFFTPoints)

numRows = size(AllRuns{1},1);
[~, numChannels] = size(AllRuns{1});

totalSamples = (numel(earlyIdx) + numel(lateIdx)) * numChannels;

XTrain = zeros(numFFTPoints, 1, 1, totalSamples, 'single');
YLabel = strings(totalSamples,1);

count = 0;

%% Early samples
for i = 1:numel(earlyIdx)

    runID = earlyIdx(i);

    for ch = 1:numChannels

        count = count + 1;

        rawSignal = AllRuns{runID}(:,ch);

        P1 = computeFFT(rawSignal, numRows);

        XTrain(:,1,1,count) = P1;

        YLabel(count) = "Early";
    end
end

%% Late samples
for i = 1:numel(lateIdx)

    runID = lateIdx(i);

    for ch = 1:numChannels

        count = count + 1;

        rawSignal = AllRuns{runID}(:,ch);

        P1 = computeFFT(rawSignal, numRows);

        XTrain(:,1,1,count) = P1;

        YLabel(count) = "Late";
    end
end

YTrain = categorical(YLabel);

end

%% ============================================================