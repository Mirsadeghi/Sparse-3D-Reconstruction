
function [R T K] = DecomPrjMat(P)
    % Camera centre
    C = -(P(:,1:3)\P(:,4));
    % QR decomposition to extract Calibration and rotation matrix 
    [R K] = qr(P(:,1:3));
    % Camera Translation 
    T = -R*C;
