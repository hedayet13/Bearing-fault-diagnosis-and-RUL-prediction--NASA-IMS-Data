%% ============================================================
%% Local Function 1: Prepare FFT Training Data
%% ============================================================

function [XTrain, YTrain] = prepareFFTTrainingData(AllRuns, earlyIdx, lateIdx, numFFTPoints)

numEarly = numel(earlyIdx);
numLate  = numel(lateIdx);

numTrain = numEarly + numLate;

L = size(AllRuns{1},1);
numChannels = size(AllRuns{1},2);

XTrain = zeros(numFFTPoints, 1, 1, numTrain*numChannels, 'single');

YLabels = strings(numTrain*numChannels,1);

count = 0;

for i = 1:numEarly
    runData = AllRuns{earlyIdx(i)};

    for ch = 1:numChannels
        count = count + 1;

        rawSignal = runData(:,ch);
        P1 = computeFFT(rawSignal, L);

        XTrain(:,:,:,count) = reshape(P1, numFFTPoints, 1, 1);
        YLabels(count) = "Healthy";
    end
end

for i = 1:numLate
    runData = AllRuns{lateIdx(i)};

    for ch = 1:numChannels
        count = count + 1;

        rawSignal = runData(:,ch);
        P1 = computeFFT(rawSignal, L);

        XTrain(:,:,:,count) = reshape(P1, numFFTPoints, 1, 1);
        YLabels(count) = "Degraded";
    end
end

YTrain = categorical(YLabels);

end
