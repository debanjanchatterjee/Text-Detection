function net = cnn_aklwdgray_init(varargin)
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

lr = [.1 2] ;

% Define network CIFAR10-quick
net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(8,8,1,96, 'single'), zeros(1, 96, 'single')}}, ...                           
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [5 5], ...
                           'stride', 5, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(2,2,96,256, 'single'), zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; 
net.layers{end+1} = struct('type', 'relu') ; % new added

% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(2,2,256,256, 'single'), zeros(1,256,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;                  
                       
% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,256,2, 'single'), zeros(1,2,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  %net = insertBnorm(net, 7) ;
end

% Meta parameters
net.meta.inputSize = [32 32 1] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end


% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
