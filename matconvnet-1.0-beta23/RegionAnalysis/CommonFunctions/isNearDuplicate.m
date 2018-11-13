function [unqCount, unqSubImHashList] = isNearDuplicate(imMean, partIm, subImSize, index, unqCount, unqSubImHashList)

    thresh = 5/16; % orig
    thresh = 7/16;
    hashSig = createHashSig(imMean, partIm, subImSize);
    
    if (0 == unqCount)        
        unqCount = unqCount + 1;
        unqSubImHashList{unqCount, 1} = index;
        unqSubImHashList{unqCount, 2} = hashSig;
        return;
    end
    
    matchFound = 0;
    for cnt = 1:unqCount
        unqImHashSig = unqSubImHashList{cnt, 2};
        allDist = pdist2(hashSig,unqImHashSig,'hamming');
        dist = diag(allDist);
        maxDist = max(dist);
        if(maxDist < thresh)
            matchFound = 1;
            break;
        end
    end
    
    if(0 == matchFound)
         unqCount = unqCount + 1;
         unqSubImHashList{unqCount, 1} = index;
         unqSubImHashList{unqCount, 2} = hashSig;            
    end
end

function [hashSig]=createHashSig(imMean, im, subImSize)
    step = 8;
    rSub = subImSize(1);
    cSub = subImSize(2);
    hashSig = zeros(16,16);
    
    hashIdx = 0;
    for rIdx = 1:step:rSub
        for cIdx = 1:step:cSub
            imCut = im(rIdx:rIdx+step-1, rIdx:rIdx+step-1);
            imSerial = reshape(imCut, [], 1)';
            imBitSerial = imSerial > imMean;
                        
            hashIdx = hashIdx + 1;
            hashElemIdx = 0;
            for idx=1:4:step*step
                hashElemIdx = hashElemIdx + 1;
                hashSig(hashIdx, hashElemIdx) = bi2de(imBitSerial(idx:idx+3));
            end
        end
    end
end