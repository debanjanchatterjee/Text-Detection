clear;
%% Just once
%cd matconvnet−1.0−beta12
%run matlab/vl_compilenn

%%
%urlwrite(...
%'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%'imagenet-vgg-f.mat') ;

%%
% setup MatConvNet
run matlab/vl_setupnn

%%
% load the pre−t7rained CNN
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;
%%
tic
% load and preprocess an image
%im = imread('peppers.png') ;
im = imread('dog.jpg');
%im = imread('map.jpg');
imtool(im);
im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
%imtool(im_);
im_ = im_ - net.meta.normalization.averageImage;
%imtool(im_);
%%
% run the CNN
res = vl_simplenn(net, im_);

%%
% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{best}, best, bestScore)) ;
toc