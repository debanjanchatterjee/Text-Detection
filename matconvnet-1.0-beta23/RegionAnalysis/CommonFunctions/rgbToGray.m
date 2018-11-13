function [im]= rgbToGray(image)
    if(size(image,3) == 3)
        im = rgb2gray(image);
    else
        im = image;
    end
end