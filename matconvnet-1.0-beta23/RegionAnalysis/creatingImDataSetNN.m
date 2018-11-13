clear;
imtool close all;
addpath('./CommonFunctions/');
%%
colorImModel = 1;
generateWaterImDb = 0;
%%
maxCandImPerType = 30000;
trainCandImPerType = 25000; % remaining are testing
testCandImPerType = maxCandImPerType - trainCandImPerType;

rDim = 32;
cDim = 32;
save('./GenData/attributes.mat', 'maxCandImPerType', 'trainCandImPerType', 'testCandImPerType', 'rDim', 'cDim');

%%
saveSubIm = 1;
numRemImToStore = 1500;


%%
if(generateWaterImDb)
    dirName = './TrainTestSamples/InDir/WaterIm/';
    if(0 == colorImModel)
        outDirName = './TrainTestSamples/OutDirGray/WaterIm';
    else        
        outDirName = './TrainTestSamples/OutDirColor/WaterIm';
    end
else
    dirName = './TrainTestSamples/InDir/NonWaterIm/';
    if(0 == colorImModel)
        outDirName = './TrainTestSamples/OutDirGray/NonWaterIm';
    else        
        outDirName = './TrainTestSamples/OutDirColor/NonWaterIm';
    end
end


%%
maxImCanGenerate = 200000;
totalSubImCount = 0;

colorDim = 1;
if(colorImModel)
    colorDim = 3;
end
subImCompleteList = uint8(zeros(rDim, cDim, colorDim, maxImCanGenerate));
candidateImList = uint8(zeros(rDim, cDim, colorDim, maxCandImPerType));

%% Code section
allFiles = dir( dirName );
allNames = { allFiles.name }; %% all files & dir

files = {allFiles(~[allFiles.isdir]).name}; %% all files
fileCount = size({allFiles(~[allFiles.isdir]).name}, 2); %% all files count

%dirs = {allFiles([allFiles.isdir]).name}; %% all dir
%dirCount = size({allFiles([allFiles.isdir]).name},2); %% all dir count

%%

for fileIndex= 1:fileCount
    fileIndex
    onlyFileName = files{fileIndex};
    startName = strcat(dirName,'/');
    fileName = strcat(startName, onlyFileName);
    
    image = imread(fileName);
    
    if(0 == colorImModel)
        im = rgb2gray(image);
    else
        im = image;
    end
    %imtool(im); 
    if(generateWaterImDb)        
        %[count, subImList] = createUniqueSubIm(im, [rDim, cDim], 12);   
        [count, subImList] = createSubImage(im, [rDim, cDim], 12);   
    else
        [count, subImList] = createSubImage(im, [rDim, cDim], 16);    
        %[count, subImList] = createUniqueSubIm(im, [rDim, cDim], 16);    
    end
    
    if(0 == count)
        disp(onlyFileName);
    else
        %subImCompleteList(:,:,totalSubImCount+1:totalSubImCount+count) = subImList(:,:,1:count);
        subImCompleteList(:,:,:,totalSubImCount+1:totalSubImCount+count) = subImList(:,:,:,1:count);
        totalSubImCount = totalSubImCount + count            
    end
end
totalSubImCount

%%
% Generate unique random no between the limit
% Block for candidate images
randomIndices = randperm(totalSubImCount, maxCandImPerType);
%candidateImList(:,:,:) = subImCompleteList(:,:,randomIndices);
candidateImList(:,:,:,:) = subImCompleteList(:,:,:,randomIndices);

% Block for remaining images those will be used for testing later
remImIndices = setdiff(1:totalSubImCount, randomIndices);
remImCount = size(remImIndices,2);
anotherRandomIndices = randperm(remImCount, numRemImToStore);
actualRemInIndices = remImIndices(anotherRandomIndices);
%remImList(:,:,:) = subImCompleteList(:,:,actualRemInIndices);
remImList(:,:,:,:) = subImCompleteList(:,:,:,actualRemInIndices);

%% Save sub im
if(saveSubIm)     
    %Store only specified remaning image
    for idx = 1:numRemImToStore
        filename = sprintf('%s/file%d.jpg',outDirName, idx);
        %imwrite(remImList(:,:,idx),filename);
        imwrite(remImList(:,:,:,idx),filename);
    end    

end
%%
%{
%for i= 1:1000:maxCandidateIm
    imX = candidateImList(:,:,8001);
    imtool(imX);
    imY =   reshape(imX,[],1)';
    imZ = reshape(imY',32,32,1,[]);
    imtool(imZ);
    %imP = permute(imZ, [2 1 3 4]);
    %imtool(imP);
%end
%}
if(generateWaterImDb)
    waterImSerialList = reshape(candidateImList,[],maxCandImPerType)';
    if(0 == colorImModel)
        save('./GenData/waterImGray.mat', 'waterImSerialList');
    else
        save('./GenData/waterImColor.mat', 'waterImSerialList');
    end
else
    nonWaterImSerialList = reshape(candidateImList,[],maxCandImPerType)';   
    if(0 == colorImModel)
        save('./GenData/nonWaterImGray.mat', 'nonWaterImSerialList');
    else
        save('./GenData/nonWaterImColor.mat', 'nonWaterImSerialList');
    end

end


