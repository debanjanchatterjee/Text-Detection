function[feature] = cnn_feature_reproducep(net, data, layer_no)
%from ISI
if(nargin < 3)
    layer_no = length(net.layers);
end

if(layer_no > length(net.layers))
    fprintf('ERROR: requered layer is outside the network\n');
    feature = [];
else
    num_data = size(data, 4);
    res = vl_simplenn(net, data(:,:,:,1)) ;
    
    scores = squeeze(gather(res(layer_no).x)) ; %changes
    feat_dim = length(scores(:));
    feature = zeros(feat_dim, num_data);
    
    for nd = 1:num_data
        nd
        res = vl_simplenn(net, data(:,:,:,nd)) ;
        
        scores = squeeze(gather(res(layer_no).x)) ; %changes
        feature(:,nd) = scores(:);
    end
end
