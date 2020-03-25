% This is a wrapper to call the CORFContourDetection from within the
% python script. 
% The communication between python and MATLAB is performed via loading and
% writing .mat files.

function [binarymap, corfresponse] = contour_detection_from_python(img, sigma, beta, inhibitionFactor, highthresh)

% dbstop in contour_detection_from_python at 8

% default arguments
if nargin == 0
    img = imread("D:\GitCode\fashion-mnist\data\cache\clean_images\00013.png");
    sigma = 1;
    beta = 4;
    inhibitionFactor = 1.8;
    highthresh = 0.007;
end

% w = 4;
rf = 3;
img1 = imresize(img,rf);
% img = padarray(img, [w,w]);
img2 = double(img1)./255;
% img2 = imnoise(img2,'gaussian',0.1);

% Evaluate 
[~, corfresponse] = CORFContourDetection(img2, sigma, beta, inhibitionFactor, highthresh);
[binarymap, ~] = CORFContourDetection(img, sigma, beta, inhibitionFactor, highthresh);

% binarymap = binarymap(w+1:end-w, w+1:end-w);
% corfresponse = corfresponse(w+1:end-w, w+1:end-w); 
corfresponse = imresize(corfresponse,1/rf);
corfresponse = (corfresponse - min(corfresponse(:)))./ (max(corfresponse(:)) - min(corfresponse(:)));


% figure;
% subplot(1,2,1);imagesc(imresize(img2,1/rf));
% subplot(1,2,2);imagesc(corfresponse);
