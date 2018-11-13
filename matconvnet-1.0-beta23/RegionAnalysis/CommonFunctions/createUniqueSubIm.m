function [unqCount, unqSubImList] = createUniqueSubIm(inIm, subImSize, stepLen)    
    [count, subImList] = createSubImage(inIm, subImSize, stepLen);
         
    unqCount = 0;
    unqSubImHashList = {};
    
    imMean = mean2(inIm);
    for cnt = 1:count
        partIm = subImList(:,:,:,cnt);
        [unqCount, unqSubImHashList] = isNearDuplicate(imMean, partIm, subImSize, cnt, unqCount, unqSubImHashList);
    end
    
    unqIndex = [unqSubImHashList{:,1}];
    unqSubImList = subImList(:,:,:,unqIndex);
end

