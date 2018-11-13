clear;

%%
% setup MatConvNet
run ./matlab/vl_setupnn

netMat = load('./TextAnalysis/GenData/netAkltd.mat');
net = vl_simplenn_tidy(netMat.net) ;
%%
trainBatch1 = load('./data/akltd/data_batch_1.mat');
trainBatch2 = load('./data/akltd/data_batch_2.mat');
trainBatch3 = load('./data/akltd/data_batch_3.mat');
trainBatch4 = load('./data/akltd/data_batch_4.mat');
trainBatch5 = load('./data/akltd/data_batch_5.mat');

testBatch = load('./data/akltd/test_batch.mat');

% Too tired, doing it bit manually

%%
total60KInput = zeros(60000, 1024);
total60KLabels = zeros(60000, 1);
total60KInput(1:10000, :) = trainBatch1.data;
total60KLabels(1:10000, :) = trainBatch1.labels;
total60KInput(10001:20000, :) = trainBatch2.data;
total60KLabels(10001:20000, :) = trainBatch2.labels;
total60KInput(20001:30000, :) = trainBatch3.data;
total60KLabels(20001:30000, :) = trainBatch3.labels;
total60KInput(30001:40000, :) = trainBatch4.data;
total60KLabels(30001:40000, :) = trainBatch4.labels;
total60KInput(40001:50000, :) = trainBatch5.data;
total60KLabels(40001:50000, :) = trainBatch5.labels;
total60KInput(50001:60000, :) = testBatch.data;
total60KLabels(50001:60000, :) = testBatch.labels;

%%

data = reshape(total60KInput',32, 32, 1, []);
%data = data - netMat.net.meta.dataMean;
data = bsxfun(@minus, data, net.meta.dataMean);

z = reshape(data,[],60000) ;
z = bsxfun(@minus, z, mean(z,1)) ;
n = std(z,0,1) ;
z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  %data = reshape(z, 32, 32, 3, []) ;
data = reshape(z, 32, 32, 1, []) ;

%%
lastButOneLayerNo = length(net.layers) - 3;
total60KFeatures =  cnn_feature_reproduce(net, data, lastButOneLayerNo);
%%
targetDir = './TextAnalysis/GenData/';
featDim = size(total60KFeatures,1);


fileName = 'trainCnnSvnFeat.txt';
completeFileName = fullfile(targetDir, fileName);

fp = fopen(completeFileName,'w');
for idx = 1:2:50000    
     idx
     fprintf(fp, '%d ', total60KLabels(idx)+1);     
     for featNum = 1:featDim 
         fprintf(fp, '%d:%f ', featNum, total60KFeatures(featNum, idx));
     end
     fprintf(fp, '\n');
end
fclose(fp);


fileName = 'testCnnSvnFeat.txt';
completeFileName = fullfile(targetDir, fileName);

fp = fopen(completeFileName,'w');
for idx = 50001:2:60000
     idx
     fprintf(fp, '%d', total60KLabels(idx)+1);
     
     for featNum = 1:featDim 
         fprintf(fp, ' %d:%f', featNum, total60KFeatures(featNum, idx));
     end
     fprintf(fp, '\n');
end
fclose(fp);


