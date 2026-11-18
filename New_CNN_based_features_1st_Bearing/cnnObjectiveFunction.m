%% ============================================================
%% Local Function 3: Bayesian Optimization Objective Function
%% ============================================================

function objective = cnnObjectiveFunction(optVars, XTrain, YTrain, numFFTPoints)

cv = cvpartition(YTrain,'HoldOut',0.2);

idxTrain = training(cv);
idxVal   = test(cv);

XTr = XTrain(:,:,:,idxTrain);
YTr = YTrain(idxTrain);

XVal = XTrain(:,:,:,idxVal);
YVal = YTrain(idxVal);

layers = createFFTCNNLayers( ...
    numFFTPoints, ...
    optVars.FeatureDim, ...
    optVars.NumFilters1, ...
    optVars.NumFilters2, ...
    optVars.NumFilters3);

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',optVars.MiniBatchSize, ...
    'InitialLearnRate',optVars.InitialLearnRate, ...
    'L2Regularization',optVars.L2Regularization, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','none');

try
    netTemp = trainNetwork(XTr, YTr, layers, options);

    YPred = classify(netTemp, XVal);

    valAccuracy = mean(YPred == YVal);

    objective = 1 - valAccuracy;

catch ME
    warning('Training failed during Bayesian optimization: %s', ME.message);
    objective = 1;
end

end