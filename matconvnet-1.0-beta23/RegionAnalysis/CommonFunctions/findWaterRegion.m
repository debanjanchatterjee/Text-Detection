function [paramList] = findWaterRegion(im, net, scale, doPadding, stride)
    maxPopulation = (32/stride)^2;
    str = sprintf('Detecting using scale %f', scale);
    disp(str);
    
    [origRow, origCol, ~] = size(im);
    
    rSub = net.meta.inputSize(1);
    cSub = net.meta.inputSize(2);
    dim = net.meta.inputSize(3);
    
    padSize = rSub; %/2;
    
    if (scale ~= 1)
        im = imresize(im, scale);
    end
    
    if doPadding
        im = padarray(im, [padSize, padSize]);
    end
    
   
    [tempRows, tempCols, ~] = size(im);
    initialGrayCompImTemp = zeros([tempRows, tempCols]); % must not go inside doPadding check
    [rIm,cIm, ~] = size(im);   
    
    for row = 1:stride:rIm-rSub+1
        for col = 1:stride:cIm-cSub+1                        
            subIm = im(row:row+rSub-1, col:col+cSub-1, :);            
                        
            im_ = double(subIm);
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
            
            %if(2 == best)                
            %    winPixsVal = initialGrayCompImTemp(row:row+rSub-1, col:col+cSub-1, :);
            %    initialGrayCompImTemp(row:row+rSub-1, col:col+cSub-1, :) = winPixsVal + bestScore; %1;
            %end
            
            winPixsVal = initialGrayCompImTemp(row:row+rSub-1, col:col+cSub-1, :);
            initialGrayCompImTemp(row:row+rSub-1, col:col+cSub-1, :) = winPixsVal + scores(2);

            
        end % end of inner for
    end % end of outer for
    
    if doPadding
        im = im(padSize+1:end-padSize,padSize+1:end-padSize);
        initialGrayCompImTemp = initialGrayCompImTemp(padSize+1:end-padSize,padSize+1:end-padSize);
    end
    
    if (scale ~= 1) % perfect
        initialGrayCompImTemp = imresize(initialGrayCompImTemp, [origRow, origCol]);
    end
    
    % new lines    
    initialGrayCompIm = uint8(initialGrayCompImTemp);
    consResponseMap = initialGrayCompImTemp/maxPopulation;
    initialCtGrayIm = uint8(consResponseMap*255);    
    initialBwCompIm = uint8(initialGrayCompIm > 0);
    
    paramList = {initialBwCompIm, consResponseMap, initialCtGrayIm};
        
end
