% This is a wrapper to call the CORFContourDetection from within the
% python script. 
% The communication between python and MATLAB is performed via loading and
% writing .mat files.

function [binarymap, corfresponse] = contour_detection_from_python(sigma, beta, inhibitionFactor, highthresh)

w = 4;

% Load the .mat inputs
img = load("./../data/cache/input_image.mat").img;
img1 = padarray(img, [w,w]);

% Evaluate 
[binarymap, corfresponse] = CORFContourDetection(img1, sigma, beta, inhibitionFactor, highthresh);

binarymap = binarymap(w+1:end-w, w+1:end-w);
corfresponse = corfresponse(w+1:end-w, w+1:end-w); 


% Save the outputs as .mat files
% save("./../data/cache/output_binary_map.mat", "binarymap")
% save("./../data/cache/output_corf_response.mat", "corfresponse")