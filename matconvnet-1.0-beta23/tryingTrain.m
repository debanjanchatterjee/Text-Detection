clear;

%%
addpath('./examples')

%%


trainingOption = 3; % 0: mnist, 1: cifar, 2: akltd

switch trainingOption
    case 0
        disp('MNIST')
        addpath('./examples/mnist')
        [netMnist, infoMnist] = cnn_mnist();
        save('netMnist.mat', 'netMnist');
    case 1
        disp('CIFAR')
        addpath('./examples/cifar')
        [net, info] = cnn_cifar();
        save('netCifar.mat', 'net');
    case 2
        disp('AKLWDGRAY')
        addpath('./examples/aklwd')
        [net, info] = cnn_aklwdgray();
        % this is needed
        net.layers{1,end}.type = 'softmax';
        save('./RegionAnalysis/GenData/netAklWdGray2.mat', 'net');
    case 3
        disp('AKLWDCOLOR')
        addpath('./examples/aklwd')
        [net, info] = cnn_aklwdcolor();
        % this is needed
        net.layers{1,end}.type = 'softmax';
        save('./RegionAnalysis/GenData/netAklWdColor.mat', 'net');
    case 4
        disp('CIFARWD')
        addpath('./examples/cifarwd')
        [net, info] = cnn_cifarwd();
        net.layers{1,end}.type = 'softmax';
        save('./RegionAnalysis/GenData/netCifarWD.mat', 'net');
    otherwise
        disp('other value')
end