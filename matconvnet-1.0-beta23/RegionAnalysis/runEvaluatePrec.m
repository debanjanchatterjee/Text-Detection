clear;
imtool close all;
%%
% add path section 
addpath('./CommonFunctions/');

%%
% setup MatConvNet
run ../matlab/vl_setupnn

%%
evaluateWater = 1;
%%
useColorModel = 1;
%%
% load the pre−trained CNN
if(0 == useColorModel)
    netMat = load('./GenData/netAklWdGray2.mat');    
else
    netMat = load('./GenData/netAklWdColor.mat');
    %netMat = load('./GenData/netCifarWD.mat');
end
net = vl_simplenn_tidy(netMat.net) ;

%%
leastScore = 1;
tic
% load and preprocess an image
if(evaluateWater)
    expectedVerdict = 2;
    if(0 == useColorModel)
        inDir = './TrainTestSamples/OutDirGray/WaterIm/';
    else
        inDir = './TrainTestSamples/OutDirColor/WaterIm/';
    end
else
    expectedVerdict = 1;
    if(0 == useColorModel)
        inDir = './TrainTestSamples/OutDirGray/NonWaterIm/';
    else
        inDir = './TrainTestSamples/OutDirColor/NonWaterIm/';
    end
end

allFiles = dir( inDir );
allNames = { allFiles.name }; %% all files & dir

files = {allFiles(~[allFiles.isdir]).name}; %% all files
fileCount = size({allFiles(~[allFiles.isdir]).name}, 2); %% all files count

rSub = net.meta.inputSize(1);
cSub = net.meta.inputSize(2);
dim = net.meta.inputSize(3);

tic
successCount = 0;
for fileIndex= 1:fileCount
    %fileIndex
    onlyFileName = files{fileIndex};
    startName = strcat(inDir,'/');
    fileName = strcat(startName, onlyFileName);
    
    image = imread(fileName);
    if(0 == useColorModel)
        im = rgbToGray(image);
    else
        im = image;
    end
    %imtool(im);
    
    %im_ = imresize(double(im), net.meta.inputSize(1:2));
    im_ = double(im);
    im_ = im_ - net.meta.dataMean;
         
    if net.meta.contNorm
        z = reshape(im_,[],1) ;                        
        z = bsxfun(@minus, z, mean(z,1)) ;
        n = std(z,0,1) ;
        z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;        
        im_ = reshape(z, rSub, cSub, dim, []) ;
    end
    
    %{
    if net.meta.whitenData
        z = reshape(im_,[],1) ;
        V = net.meta.whitenInfo.V;
        d2 = net.meta.whitenInfo.d2;
        en = net.meta.whitenInfo.en;
        z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
        im_ = reshape(z, rSub, cSub, dim, []) ;
    end
    %}

    
    % run the CNN
    res = vl_simplenn(net, im_);


    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores);
    
    %figure(1) ; clf ; imagesc(image) ;
    %title(sprintf('%s (%d), score %.3f',...
    %    net.meta.classes.name{best}, best, bestScore)) ;
    if(best == expectedVerdict)
        successCount = successCount + 1;
        if(leastScore > bestScore)
            leastScore = bestScore;
        end
    else
        imtool(im);
        imtool close all;
    end
end
toc
successCount
leastScore
str = sprintf('Success Rate : %f', successCount*100/fileCount)



