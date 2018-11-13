clear;
%imtool close all;

%%
% add path section 
addpath('./CommonFunctions/');
%addpath('../../SceneTextCNN_demo/detectorDemo/');
%load('./GenData/theTextGTDb.mat');
%%
% setup MatConvNet
run ../matlab/vl_setupnn

%%
stride = 8; % 4 or 8. Recommended 4 


doPadding = 1;
confidenceFactor = 0.3;
useColorModel = 1;
%%
% load the pre−trained CNN
if(0 == useColorModel)    
    netMat = load('./GenData/netAklWdGray.mat'); % either this one
    
else    
    netMat = load('./GenData/netAklWdColor.mat');
    %netMat = load('./GenData/netCifarWD.mat');
end
net = vl_simplenn_tidy(netMat.net) ;

rSub = net.meta.inputSize(1);
cSub = net.meta.inputSize(2);
dim = net.meta.inputSize(3);
%%
tic

fileName = './Images/Input.jpg';

image = imread(fileName);

if(0 == useColorModel)
    im = rgbToGray(image);
else
    im = image;
end
imtool(im);
%im = imadjust(im); % it was just a try
%imtool(im);

%%
paramList = findWaterRegion(im, net, 1, doPadding, stride);
initialBwCompIm = paramList{1};
initialGrayCompIm = paramList{2};
initialCtGrayIm = paramList{3};
imtool(initialGrayCompIm);
%imtool(initialCtGrayIm);

%imtool(logical(initialBwCompIm));
%imtool(im.*initialBwCompIm);

