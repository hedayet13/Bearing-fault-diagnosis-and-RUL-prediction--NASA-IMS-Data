%% ============================================================
%% Local Function 4: Create FFT-CNN Layers
%% ============================================================

function layers = createFFTCNNLayers( ...
    numFFTPoints, featureDim, numFilters1, numFilters2, numFilters3)

layers = [

imageInputLayer([numFFTPoints 1 1], ...
'Name','input', ...
'Normalization','none')

convolution2dLayer([64 1],numFilters1, ...
'Padding','same', ...
'Name','conv1')

batchNormalizationLayer('Name','bn1')
reluLayer('Name','relu1')

maxPooling2dLayer([4 1], ...
'Stride',[4 1], ...
'Name','pool1')

convolution2dLayer([32 1],numFilters2, ...
'Padding','same', ...
'Name','conv2')

batchNormalizationLayer('Name','bn2')
reluLayer('Name','relu2')

maxPooling2dLayer([4 1], ...
'Stride',[4 1], ...
'Name','pool2')

convolution2dLayer([16 1],numFilters3, ...
'Padding','same', ...
'Name','conv3')

batchNormalizationLayer('Name','bn3')
reluLayer('Name','relu3')

maxPooling2dLayer([4 1], ...
'Stride',[4 1], ...
'Name','pool3')

convolution2dLayer([8 1],128, ...
'Padding','same', ...
'Name','conv4')

batchNormalizationLayer('Name','bn4')
reluLayer('Name','relu4')

globalAveragePooling2dLayer('Name','gap')

fullyConnectedLayer(featureDim, ...
'Name','cnn_features')

reluLayer('Name','feature_relu')

fullyConnectedLayer(2, ...
'Name','fc_output')

softmaxLayer('Name','softmax')

classificationLayer('Name','classification')
];

end