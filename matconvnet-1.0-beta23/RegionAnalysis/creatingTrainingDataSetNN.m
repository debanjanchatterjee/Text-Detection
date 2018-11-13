clear;
colorImModel = 1;
%%
% load section
load('./GenData/attributes.mat');
if(0 == colorImModel)
    load('./GenData/nonWaterImGray.mat');
    load('./GenData/waterImGray.mat');
else
    load('./GenData/nonWaterImColor.mat');
    load('./GenData/waterImColor.mat');
end

%%
if(0 == colorImModel)
    genDir = '../data/aklWdGray/';
else
    genDir = '../data/aklWdColor/';
end

%%
label_names = cell(2,1);
label_names{1} = 'nonwater';
label_names{2} = 'water';
filePath = strcat(genDir, 'batches.meta.mat');
save(filePath, 'label_names');

%
numDataPerFile = 10000; 
%%
% Create 6 files : 5 for training, 1 for testing

nonWaterTraining = nonWaterImSerialList(1:trainCandImPerType,:);
waterTraining = waterImSerialList(1:trainCandImPerType,:);

nonWaterTesting = nonWaterImSerialList(trainCandImPerType+1:maxCandImPerType,:);
waterTesting = waterImSerialList(trainCandImPerType+1:maxCandImPerType,:);


%% Create Training data *************************************

numTrainingData = 2*trainCandImPerType;

% Concat Text Training and NonText 
trainingData = cat(1, nonWaterTraining, waterTraining);
trainingLabels = uint8(zeros(numTrainingData, 1));
trainingLabels(trainCandImPerType+1:numTrainingData,1) = 1;
% = 'training batch 1 of 5';

% Just shake well - generate unique random indices between 1..50K
randomIndices = randperm(numTrainingData, numTrainingData)';
trainingDataMixed = trainingData(randomIndices,:);
trainingLabelsMixed = trainingLabels(randomIndices);

iterCount = 1;
for idx=1:numDataPerFile:numTrainingData    
    data = trainingDataMixed(idx:iterCount*numDataPerFile,:);
    labels = trainingLabelsMixed(idx:iterCount*numDataPerFile);
    batch_label = sprintf('training batch %d of 5',iterCount);
    fileName = sprintf('data_batch_%d.mat',iterCount);
    filePath = strcat(genDir, fileName);    
    save(filePath, 'data', 'labels', 'batch_label');
    iterCount = iterCount + 1;
end
%% Create Testing data *************************************

numTestingData = 2*testCandImPerType;

% Concat Testing Text and NonText 
testingData = cat(1, nonWaterTesting, waterTesting);
testingLabels = uint8(zeros(numTestingData, 1));
testingLabels(testCandImPerType+1:numTestingData,1) = 1;

% Just shake well - generate unique random indices between 1..10K
randomIndices = randperm(numTestingData, numTestingData)';
testingDataMixed = testingData(randomIndices,:);
testingLabelsMixed = testingLabels(randomIndices);

data = testingDataMixed;
labels = testingLabelsMixed;
batch_label = 'testing batch 1 of 1';
fileName = 'test_batch.mat';
filePath = strcat(genDir, fileName);    
save(filePath, 'data', 'labels', 'batch_label');

