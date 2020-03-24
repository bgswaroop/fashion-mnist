% This is a wrapper to call the CORFContourDetection from within the
% python script. 
% The communication between python and MATLAB is performed via loading and
% writing .mat files.

function [binarymap, corfresponse] = contour_detection_from_python(img, sigma, beta, inhibitionFactor, highthresh)

% dbstop in contour_detection_from_python at 8

% default arguments
if nargin == 0
    sigma = 1;
    beta = 4;
    inhibitionFactor = 1.8;
    highthresh = 0.007;
end

% w = 4;
% img = imread("D:\GitCode\fashion-mnist\data\cache\clean_images\00001.png");
% img1 = padarray(img, [w,w]);
% img1 = double(img1);
% img2 = imnoise(img1,'gaussian',0.1);

% Evaluate 
[binarymap, corfresponse] = CORFContourDetection(img, sigma, beta, inhibitionFactor, highthresh);

corfresponse = corfresponse ./ max(corfresponse(:));
% binarymap = binarymap(w+1:end-w, w+1:end-w);
% corfresponse = corfresponse(w+1:end-w, w+1:end-w); 


% Save the outputs as .mat files
% save("./../data/cache/output_binary_map.mat", "binarymap")
% save("./../data/cache/output_corf_response.mat", "corfresponse")