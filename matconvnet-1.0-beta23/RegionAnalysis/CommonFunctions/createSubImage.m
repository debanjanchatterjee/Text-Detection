function [count, subImList] = createSubImage(inIm, subImSize, stepLen)
    listLen = 5000;
    rSub = subImSize(1);
    cSub = subImSize(2);
    [rIm,cIm,dim] = size(inIm);
    
    subImList = uint8(zeros(rSub, cSub, dim, listLen));
    count = 0;
    
    if(rIm < rSub) || (cIm < cSub)
        disp('small size im')
        return;
    end

    for row = 1:stepLen:rIm-rSub+1
        for col = 1:stepLen:cIm-cSub+1
            
            count = count+1;
            subIm = inIm(row:row+rSub-1, col:col+cSub-1,:);
            subImList(:,:,:,count) = subIm;
            %imtool(subIm);
            
            if(count == listLen)
                disp('list exausted')
                return;
            end
        end
    end
end