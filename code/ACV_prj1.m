function ACV_prj1
time_dur = 0;
txt{1,1} = 'Elapsed Time'; tic
imgPath = '..\dataset\images\'; dCell = dir([imgPath '*.jpg']);
disp('Stage 1 : Loading Stereo image from disk');
Img = cell(1,length(dCell));

for d = 1:length(dCell)
    Img{d} = imread([imgPath dCell(d).name]);
end

im1 = Img{1} ;  im2 = Img{2} ;
if size(im1,3) > 1 , im1g = rgb2gray(im1);else im1g = im1 ; end
if size(im2,3) > 1 , im2g = rgb2gray(im2);else im2g = im2 ; end
im1g = im2single(im1g); 
im2g = im2single(im2g);

tmp_time = toc;
time_dur = time_dur + tmp_time;
txt{1,2} = tmp_time;
disp(txt)
% --------------------------------------------------------------------
%                                                         SIFT matches
% --------------------------------------------------------------------
disp('Stage 2 : Computing matches between two consecutive frames');
tic
% Extracting SIFT features from images
[f1,d1] = vl_sift(im1g,'EdgeThresh',20) ;
[f2,d2] = vl_sift(im2g,'EdgeThresh',20) ;
% Match descriptor of features to find correspondences
[matches, ~] = vl_ubcmatch(d1,d2,1.75);
numMatches = size(matches,2);
% Stack matched points
X1 = [f1(1,matches(1,:));f1(2,matches(1,:))];
X2 = [f2(1,matches(2,:));f2(2,matches(2,:))];
% Compute Elapsed Time for this stage
tmp_time = toc;
time_dur = time_dur + tmp_time;
txt{1,2} = tmp_time;  disp(txt)

% --------------------------------------------------------------------
%                                         Computing Fundemental Matrix
% --------------------------------------------------------------------
disp('Stage 3 : Computing Fundemental Matrix using RANSAC');
tic
% Augmenting matched points with element "1".
x1 = [X1;ones(1,numMatches)];
x2 = [X2;ones(1,numMatches)];
% Find Fundamental matrix by RANSAC
[F idx] = RansacF(x1,x2);
% Compute Elapsed Time for this stage
tmp_time = toc;
time_dur = time_dur + tmp_time;
txt{1,2} = tmp_time;  disp(txt)

% --------------------------------------------------------------------
%                                                         Rectification
% --------------------------------------------------------------------   
disp('Stage 4 : Compute Rectification Transform');
tic

[t1 t2] = ImgRectify(F,X1(:,idx)',X2(:,idx)',size(im2g));
T1 = maketform('projective',t1);
T2 = maketform('projective',t2);
[IR1 , x_dta1 , y_dta1] = imtransform(im1,T1);
[IR2 , x_dta2 , y_dta2] = imtransform(im2,T2);

IR1 = rgb2gray(IR1);
IR2 = rgb2gray(IR2);
% Rectify images based on calculated transforms
[IcR1 IcR2] = ImTrans(im2double(im1),t1,im2double(im2),t2);
Itmp11 = rgb2gray(IcR1);    Itmp22 = rgb2gray(IcR2);
% Coefficient to resize the images in order to speed up the computation
co = 4;
NCC1 = normxcorr2(imresize(Itmp11,1/co),imresize(IR1,1/co));
NCC2 = normxcorr2(imresize(Itmp22,1/co),imresize(IR2,1/co));
NCC1 = NCC1 == max(max(NCC1));
[rows cols] = size(NCC1);
Y_Idx = kron(1:cols,ones(rows,1));
X_Idx = kron((1:rows)',ones(1,cols));
C1 = [X_Idx(NCC1) Y_Idx(NCC1)]*co;

NCC2 = NCC2 == max(max(NCC2));
[rows cols] = size(NCC2);
Y_Idx = kron(1:cols,ones(rows,1));
X_Idx = kron((1:rows)',ones(1,cols));
C2 = [X_Idx(NCC2) Y_Idx(NCC2)]*co;
St1 = C1 - [size(IcR1,1) size(IcR1,2)];
St2 = C2 - [size(IcR2,1) size(IcR2,2)];
% Compute Elapsed Time for this stage
tmp_time = toc;
time_dur = time_dur + tmp_time;
txt{1,2} = tmp_time;  disp(txt)

% --------------------------------------------------------------------
%                                                         Dense Matching
% --------------------------------------------------------------------  
disp('Stage 6 : Dense Matching');

tic
pts1 = detectSURFFeatures(rgb2gray(IcR1),'MetricThreshold',100);
tmp_pts = round(pts1(:,:).Location)';
P1 = [tmp_pts(2,:);tmp_pts(1,:)];
P2 = P1*0;
[rows cols] = size(IcR2);
% Making Index matrix for "X" and "Y" location. this allow us to speed up
% index recovery operation
Y_Idx = kron(1:cols,ones(rows,1));
X_Idx = kron((1:rows)',ones(1,cols));
% Make temporary images to use in the NCC operation
I1gray = double(rgb2gray(IcR1));
I2gray = double(rgb2gray(IcR2));
[~,C1] = size(I1gray);
[~,C2] = size(I2gray);
% index counter
k = 0;
% This loop choose one point from lft frame and try to find its
% correspondence in the right frame in the rectified image. so the search
% location limited to one row. becuase of error i recommend that search
% should be in ares cwhich contain some rows up and bottom of the chooses
% row.
for s = P1
    % increase index counter     
    k = k + 1;
    % select row and Column index    
    i = s(1);  
    j = s(2);
    % Create a blank index image for X,Y coordinate recovery      
    im_idx = logical(I2gray * 0);
    % Evaluate near edge points
    if (j-7) >= 1 && (j+7) <= cols && (i-7)>=1 && (i+7) <=rows
        % select a template around the pixel in the first frame         
        Template = I1gray(i-5:i+5,j-5:j+5);
        % NCC operation cannot be done on a blank part of image
        if std2(Template) ~= 0
            % select Region Of Interest in the second frame as some rows 
            % up and bottom of current row.
            RoI = I2gray(i-7:i+7,:);
            % NCC operation on the template and Region of Interest in the 
            % second frame.
            tmp = normxcorr2(Template,RoI);
            % select peak value of NCC operation             
            val = max(max(tmp));            
            D = (size(Template,1)-1)/2;
            % make index matrix which shows the location of the peak value 
            % of the NCC operation. 
            ad = (tmp(D+1:end-D,D+1:end-D) == val);
            % We select correspondance in the second frame if two condition
            % satisfied.first, if we found one correspondence.second, if
            % the result of NCC operation is greater than threshold.
            if sum(sum(ad)) == 1
                im_idx(i-7:i+7,:) = ad;
                loc = [X_Idx(im_idx) Y_Idx(im_idx)];
                sgn = ((j <= (C1)/2) & (loc(2) <= (C2/2))) | ((j > (C1)/2) & (loc(2) > (C2/2)));
                if val >= 0.8 && sgn == 1               
                    P2(:,k) = loc';          
                end
            end
        end
    end
end

tmp1 = (sum(P1 == 0) > 0) ;
P1(:,tmp1) = [];P2(:,tmp1) = [];
tmp2 = (sum(P2 == 0) > 0) ;
P1(:,tmp2) = [];P2(:,tmp2) = [];

Ptt1(1,:) = P1(1,:)+St1(1)+y_dta1(1);  Ptt1(2,:) = P1(2,:)+St1(2)+x_dta1(1);
Ptt2(1,:) = P2(1,:)+St2(1)+y_dta2(1);  Ptt2(2,:) = P2(2,:)+St2(2)+x_dta2(1);
% Change "x" and "y" element to be compatible with matlab notation for
% computing transform coordinates
Ptt1 = [Ptt1(2,:);Ptt1(1,:);ones(1,size(Ptt1,2))];
Ptt2 = [Ptt2(2,:);Ptt2(1,:);ones(1,size(Ptt2,2))];
% Inverse Transform of Rectified space to Original space
Pi1 = (t1')\Ptt1;    Pi2 = (t2')\Ptt2;
% Chnage to original notation
Pi1 = [Pi1(2,:);Pi1(1,:);Pi1(3,:)];
Pi2 = [Pi2(2,:);Pi2(1,:);Pi2(3,:)];
% Normalization with respect to last element
temp1 = kron([1 1 1]',Pi1(3,:));    temp2 = kron([1 1 1]',Pi2(3,:));
Pi1 = round(Pi1 ./ temp1);  Pi2 = round(Pi2 ./ temp2);

% 3D Structure Reconstruction
% Projection matrix between images 1 and 2
PMat1 = [ 6691.06 747.814 -573.702 -339145; -197.544 -2173.34 -6397.91 4.673e+06; -0.0127641 0.89847 -0.438849 2658.3];
PMat2 = [ 4200.51 5257.67 -609.352 -1.07298e+06; 1378.44 -1704.78 -6394.47 3.94252e+06; -0.645659 0.624306 -0.439734 2977.41];
% Reconstruct 3D structure as the least-squares solution of a system of linear equations 
Xp = PointRec(Pi1,Pi2,PMat1,PMat2);
% MDepth = mean(Xp(3,:));
% XpIdx = Xp(3,:) > MDepth;
% Xp(:,XpIdx)=[];
% Creat Mesh from 3D points
mesh = pointCloud2mesh(Xp(1:3,:)');
% Make PLY file to show on MeshLab
makePly(mesh,'Output2.ply')
% Compute Elapsed Time for this stage
tmp_time = toc;
time_dur = time_dur + tmp_time;
txt{1,2} = tmp_time;  disp(txt)
% Total Elapsed Time
txt{1,1} = 'Total time';
txt{1,2} = time_dur; disp(txt)

% --------------------------------------------------------------------
%                                                         Show Results
% --------------------------------------------------------------------
disp('Stage 7 : Show results');

% ROI in the Rectified images with result of dense matching
figure(1);clf
subplot 211
dcr1 = max(size(IcR2,1)-size(IcR1,1),0) ;
dcr2 = max(size(IcR1,1)-size(IcR2,1),0) ;
IcR = ([padarray(IcR1,dcr1,'post') padarray(IcR2,dcr2,'post')]) ;
imshow(IcR);    o = size(IcR1,2);   hold on
scatter(P1(2,:)   , P1(1,:))
scatter(P2(2,:)+o , P2(1,:))
title('ROI in Rectified Images')

% Final Result
subplot 212
dh1 = max(size(im2,1)-size(im1,1),0) ;
dh2 = max(size(im1,1)-size(im2,1),0) ;
I = ([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
imshow(I);  hold on;    o = size(im1,2) ;
scatter(Pi1(2,:)   ,Pi1(1,:))
scatter(Pi2(2,:)+o ,Pi2(1,:))
title('Final Result')


%%                            Other Materials
% --------------------------------------------------------------------  
% --------------------------------------------------------------------


% --------------------------------------------------------------------
%                                                         Hat Operator
% --------------------------------------------------------------------

function out = HatOpt(in,c)
% c is a condition parameter, if c==1 then function compute hat and if c==0
% compute inverse of hat operation
    if c == 1
        out = [  0     -in(3)   in(2);
                in(3)    0     -in(1);
               -in(2)   in(1)    0 ] ;
    elseif c == 0
        out = [in(3,2) in(1,3) in(2,1)]';
    end
    
    

% --------------------------------------------------------------------
%                          Compute F with Linear Eight Point Algorithm
% --------------------------------------------------------------------
function F = FunMatCal(x1,x2)
% This function computes the fundamental matrix from 8 or more matching 
% points in a pair of images.  The normalised 8 point algorithm 6.1 
% given by Yi Ma, Stefano Soatto, Jana Kosecka and S. Shankar Sastry on 
% page 212 "An Invitation to 3D Vision" book .

% Structure:   F = FunMatCal(x1, x2)
%
% Input arguments:
%       x1, x2 - Two sets of corresponding 3xN set of homogeneous points.
% Output argument:
%       F      - 3x3 fundamental matrix such that x2'*F*x1 = 0.

% Written by S.Ehsan Mirsadeghi - April 2013
% Department of Electrical Engineering
% Sharif University of Technology 
%
% This Software is provided without any warranty.
    %% Normalization of image coordinates
    mu1 = mean(x1');  sigma1 = std(x1');
    mu2 = mean(x2');  sigma2 = std(x2');
    H1 = [  1/sigma1(1)     0       -mu1(1)/sigma1(1) ;
                0       1/sigma1(2) -mu1(2)/sigma1(2) ;
                0           0               1        ];
    H2 = [  1/sigma2(1)     0       -mu2(1)/sigma2(1) ;
                0       1/sigma2(2) -mu2(2)/sigma2(2) ;
                0           0               1        ];
    x1 = H1 * x1;
    x2 = H2 * x2;
    
    %% Compute a first approximation of the fundamental matrix
    n = length(x1);
    for i=1:n    
        A(i,:) = kron(x1(:,i),x2(:,i))';
    end

	[~,~,V] = svd(A); 

    % Obtain Fundamental Matrix corresponding to the smallest eigen value.
    F = reshape(V(:,9),3,3);
    
    %% Impose the rank constraint and recover the fundamental matrix
    [U,S,V] = svd(F);
    Sigma = S;
    Sigma(3,3) = 0;
    F = U*Sigma*V';
    
    % Denormalization of image coordinates
    F = H2'*F*H1;


% --------------------------------------------------------------------
%                                       Fundamental Matrix with RANSAC
% -------------------------------------------------------------------- 
function [F ok] = RansacF(x1,x2,tr,itr)
% tr : threshold value
% itr: number of iteration

if size(x1,2) ~= size(x2,2)
    error('Number of matches points should be equal')
end
numMatches = size(x1,2) ;
if nargin == 2
    tr = 1;
    itr = numMatches*3;   
end
% Compute Similarity Transform With RANSAC

score = zeros(1,itr);
CP_index{1,itr} = 0;
ok_total = cell(1,itr);
d = zeros(1,size(x1,2));
% Threshold for reject outliers

e3_hat = [ 0    -1     0
           1     0     0
           0     0     0];

for t = 1:itr
    % Select 8 correspondences randomly     
    subset = vl_colsubset(1:numMatches,8);
    CP_index{t} = subset;
    % Stack parameters of similarity transform into matrix A
    F = FunMatCal(x1(:,subset),x2(:,subset));
    for i = 1:size(x1,2)
        d(i) = ((x2(:,i)'*F*x1(:,i)).^2) ./ ((norm(e3_hat*F*x1(:,i)).^2)+((norm(x2(:,i)'*F*e3_hat)).^2));
    end
    % Count number of inliers based on threshold "tr"
    ok_total{t} = (d < tr) ;
    score(t) = sum(ok_total{t});
end

% Look for transform with the largest inliers
[~, best] = max(score);
ok = ok_total{best};
% id = CP_index{best};
F = FunMatCal(x1(:,ok),x2(:,ok));
F = F/F(3,3);


% --------------------------------------------------------------------
%                                                        Rectification
% -------------------------------------------------------------------- 
function [H1,H2] = ImgRectify(F,X1,X2,SizeImg)

% Change the F matrxi to matlab form.
% in matlab location of image srepresent in the form:[row col] and row is
% belong to the y-axis, so we should change the fundamental matrix to the
% prper form. if we change the matrix as bellow we can use matlab notation
% for pixel location.

% [f1 f2 f3]         [f5 f4 f6]
% |f4 f5 f6|  ---->  |f2 f1 f3|
% [f7 f8 f9]         [f8 f7 f9]

F = [F(2,2) F(2,1) F(2,3) ;
     F(1,2) F(1,1) F(1,3) ;
     F(3,2) F(3,1) F(3,3)];

% Translate the image center to the origin.
Gt = [1     0   -SizeImg(1)/2;
      0     1   -SizeImg(2)/2;
      0     0        1      ];

[U, ~, ~] = svd(F);
% compute epipole 2 as a left null space of F
e2 = U(:, 3);

Ep = Gt * e2;
if Ep(3) < 0;Ep = -Ep;end

% Rotation around the z-axis that rotates the translated epipole onto 
% the x-axis.
% if we have epipole in the location [y x 1]' we can put it on the x-axis
% by following transformation:
% [y    -x     0      ] [y]   [      0      ]
% |x     y     0      |*|x| = |norm([x y])^2|
% [0     0 norm([x y])] [1]   [ norm([x y]) ]

Gr = [Ep(2)  -Ep(1)                 0                    ;
      Ep(1)   Ep(2)                 0                    ;
        0       0     norm(Ep(1:2))*(((Ep(2)>=0)*2)-1)   ];
Ep = Gr * Ep;

% Transforms the epipole from the x-axis in the image plane to infinity
% Note: take care for Xe which located on the second column of third row

  G = [1         0        0 ; 
       0         1        0 ; 
       0   -Ep(3)/Ep(2)   1];

% Compute the overall transformation for the second image.
T2 = G * Gr * Gt;
Gt([1,2], 3) = -Gt([1,2], 3);
T2 = Gt * T2;
% Normalization of the Transform matrix
T2 = T2 / T2(end);

% Compute three-parameter family of homographies compatible
% with the fundamental matrix F by eq. (T_hat)'*F + Tv'
M = HatOpt(e2,1)'*F + e2*rand(1,3);
% Predefined H1 with multiplying H2 with considering M as 
% initial quess of H
H1r = T2 * M;

% Compute the transformation for the first camera so the points have
% (approximately) the same column locations in the first and second images.
Ht = [H1r(2,2) H1r(2,1) H1r(2,3) ;
      H1r(1,2) H1r(1,1) H1r(1,3) ;
      H1r(3,2) H1r(3,1) H1r(3,3)]';
X1 = [X1';ones(1,size(X1',2))];
P1 = Ht'*X1;
P1 = [P1(2,:);P1(1,:);P1(3,:)];

% Restore Transform to the original form.
% [h5 h4 h6]         [h1 h2 h3]
% |h2 h1 h3|  ---->  |h4 h5 h6|
% [h8 h7 h9]         [h7 h8 h9]
H2 = [T2(2,2) T2(2,1) T2(2,3) ;
      T2(1,2) T2(1,1) T2(1,3) ;
      T2(3,2) T2(3,1) T2(3,3)]';

X2 = [X2';ones(1,size(X2',2))];
P2 = H2'*X2;
P2 = [P2(2,:);P2(1,:);P2(3,:)];
% check for very small elements
p2Infinity = abs(P2(3,:)) < eps;
P2(3,p2Infinity) = eps;

b = P2(2,:) ./ P2(3,:) .* P1(3,:);
% infer solution for vector "v" in order to minimize the disparity between 
% correspondences in two images
v = P1' \ b';
H1c = [   1     0     0  ;
         v(1)  v(2)  v(3);
          0     0     1 ];

% Rectification transform for the first view
T1 = H1c * H1r;

% Normalization of the Transform matrix  
T1 = T1 / T1(end);

% Restore Transform to the original form.
% [h5 h4 h6]         [h1 h2 h3]
% |h2 h1 h3|  ---->  |h4 h5 h6|
% [h8 h7 h9]         [h7 h8 h9]

H1 = [T1(2,2) T1(2,1) T1(2,3) ;
      T1(1,2) T1(1,1) T1(1,3) ;
      T1(3,2) T1(3,1) T1(3,3)]';


% --------------------------------------------------------------------
%                                                       Transformation
% -------------------------------------------------------------------- 
function [iT1 iT2] = ImTrans(I1,t1,I2,t2)

% Compute the transformed location of image corners.
numRows = size(I1, 1);
numCols = size(I1, 2);
inPts = [1, 1; 1, numRows; numCols, numRows; numCols, 1];
tform1 = maketform('projective', t1);
t1 = projective2d(t1);
outPts(1:4,:) = tformfwd(inPts, tform1);
numRows = size(I2, 1);
numCols = size(I2, 2);
inPts = [1, 1; 1, numRows; numCols, numRows; numCols, 1];
tform2 = maketform('projective', t2);
t2 = projective2d(t2);
outPts(5:8,:) = tformfwd(inPts, tform2);

%--------------------------------------------------------------------------
% Compute the common rectangular area of the transformed images.
xSort = sort(outPts(:,1));
ySort = sort(outPts(:,2));
bbox(1) = ceil(xSort(4));
bbox(2) = ceil(ySort(4));
bbox(3) = floor(xSort(5)) - bbox(1) + 1;
bbox(4) = floor(ySort(5)) - bbox(2) + 1;

%--------------------------------------------------------------------------
% Generate a composite made by the common rectangular area of the
% transformed images.
%{
htrans = vision.GeometricTransformer(...
  'TransformMatrixSource', 'Input port', ...
  'OutputImagePositionSource', 'Property', ...
  'OutputImagePosition', bbox);
%}
iT1 = imwarp(I1, t1, 'OutputView', imref2d(bbox([4 3])));
iT2 = imwarp(I2, t2, 'OutputView', imref2d(bbox([4 3])));


% --------------------------------------------------------------------
%                                                   3D Reconstrcution
% -------------------------------------------------------------------- 
% Reconstruction From Image Correspondence
% From page 191, two corresponding image points of the form 
% x1 = [Xl, Yl, l]T and x2 = [X2, Y2, l]T yield four linearly independent 
% constraints on Xp. The projective structure can then be recovered as 
% the least-squares solution of a system of linear equations M * Xp = 0.
% This can be done, for instance, using the SVD . As in the calibrated
% case, this linear reconstruction is suboptimal in the presence of noise.
% 
% Structure:   Xp = PointRec(x1,x2,P1,P2)
%        or    Xp = PointRec(x1,x2,P2) - with P1 = [I , 0]
% Input arguments:
%       x1, x2 - Two sets of corresponding 3xN set of homogeneous points.
%       P1, P2 - Two 3x4 Projection matrix related to the first and second
%                    camera frame.
% Output argument:
%       Xp   - Reconstructed Point in the 3D space.

function Xp = PointRec(x1,x2,P1,P2)
npts = size(x1,2);
Xp = zeros(4,npts);  
    T = [x1(1,:)'*P1(3,:) - kron(ones(npts,1),P1(1,:));
         x1(2,:)'*P1(3,:) - kron(ones(npts,1),P1(2,:));
         x2(1,:)'*P2(3,:) - kron(ones(npts,1),P2(1,:));
         x2(2,:)'*P2(3,:) - kron(ones(npts,1),P2(2,:))];
 
    for i = 1 : npts
        M = [T(i,:);T(i+npts,:);T(i+(2*npts),:);T(i+(3*npts),:)];
        [~,~,V] = svd(M'*M);
        % Normalized Reconstruction
        Xp(:,i) = V(:,end)'/V(end,end);
    end


% --------------------------------------------------------------------
%                                          Convert point cloud to Mesh
% -------------------------------------------------------------------- 
function mesh = pointCloud2mesh(data, refNormal, stdTol)

% mesh = meshD(data, refNormal, stdTol)

% Author : Ajmal Saeed Mian {ajmal@csse.uwa.edu.au}
%           Computer Science. Univ of Western Australia
%
% This function takes data points performs triangulation on it, filters out
% incorrecp polygons and outputs a mesh data structure like the newMesh
% function.
%
% Arguments : data - Nx3 vertex coordinates [x y z] of the pointcloud
%             stdTol - (optional) tolerance for edge filtering. default is 0.6
%             
%             refNormal - (optional) 1x3 vector in the sensor direction
%                         =[0 0 1] if the sensor looking towards the -z_axis
%
% Return : mesh - mesh data structure
%                       vertices: Nx3 vertex coordinates
%                       triangles: M triangles using index numbers of the vertices
%                       resolution: the mean edge length of triangles
%                       stdeviation: the standard deviation o edge lengths
%                       triangleNormals: Mx3 normal vectors of each triangle
%                       vertexNormals: Nx3 normal vectors of each vertex
%                       vertexNtriangles: Nx1 cell of neighboring triangles 
%                                           of each vertex
%                       triangleNtriangles: Mx1 cell of nieghboring triangles
%                                               of each triangle
%
% Copyright : This code is written by Ajmal Saeed Mian {ajmal@csse.uwa.edu.au}
%              Computer Science, The University of Western Australia. The code
%              may be used, modified and distributed for research purposes with
%              acknowledgement of the author and inclusion this copyright information.
%
% Disclaimer : This code is provided as is without any warrantly.

if nargin == 1
    PC = pca(data);
    data = data*PC;
    refNormal = [0 0 1];
    refNormal = refNormal * PC;
end

if nargin < 3
    stdTol = 0.6;
end

tri = delaunay(data(:,1),data(:,2));
tri(:,4) = 0; % initialize 4th column to store maximum edge length

edgeLength = [sqrt(sum((data(tri(:,1),:) - data(tri(:,2),:)).^2,2)),...
        sqrt(sum((data(tri(:,2),:) - data(tri(:,3),:)).^2,2)),...
        sqrt(sum((data(tri(:,3),:) - data(tri(:,1),:)).^2,2))];

tri(:,4) = max(edgeLength,[],2);

resolution = mean(edgeLength(:));
stdeviation = std(edgeLength(:));
filtLimit = resolution + stdTol*stdeviation;

bigTriangles = find(tri(:,4) > filtLimit); %find index numbers of triagles with edgelength more than filtLimit
tri(bigTriangles,:) = []; % remove all triangles with edgelength more than filtlimit
tri(:,4) = []; % remove the max edgeLength column

edgeLength(bigTriangles,:) = []; % remove edges belonging to triangles which are removed
edgeLength = edgeLength(:); 
resolution = mean(edgeLength); % find the mean of the remaining edges
stdeviation = std(edgeLength);

mesh = [];
if nargin < 2
    data = data*PC';% multiply the data points by the inverse PC
    refNormal = refNormal * PC';
end
mesh.vertices = data;  
mesh.triangles = tri;
mesh.resolution = resolution;
mesh.stdeviation = stdeviation;

noOfpolygons = size(tri,1);
noOfpoints = size(data,1);
mesh.triangleNormals = zeros(noOfpolygons,3); % innitialize a matrix to store polygon normals
mesh.vertexNormals = zeros(noOfpoints,3); % innitialize a matrix to store point normals
mesh.vertexNtriangles = cell(noOfpoints, 1); %a cell array to store neighbouring polygons for the current point
mesh.triangleNtriangles = cell(noOfpolygons, 1); % to store neighbors of current polygon

for ii = 1:noOfpolygons %find normals of all polygons
    %indices of the points from which the polygon is made
    pointIndex1 = mesh.triangles(ii,1);
    pointIndex2 = mesh.triangles(ii,2);
    pointIndex3 = mesh.triangles(ii,3);
    
    %coordinates of the points
    point1 = mesh.vertices(pointIndex1,:);
    point2 = mesh.vertices(pointIndex2,:);
    point3 = mesh.vertices(pointIndex3,:);
    
    vector1 = point2 - point1;
    vector2 = point3 - point2;
    
    normal = cross(vector1,vector2);
    normal = normal / norm(normal);
    
    theta = acos(dot(refNormal, normal));
    if theta > pi/2
        normal = normal * (-1);
        a = mesh.triangles(ii,2);
        mesh.triangles(ii,2) = mesh.triangles(ii,1);
        mesh.triangles(ii,1) = a;
    end
    
    mesh.triangleNormals(ii,:)=normal;   
            
    %make entry of this polygon as the neighbouring polygon of the three
    %vertex points    
    mesh.vertexNtriangles(pointIndex1,1)={[mesh.vertexNtriangles{pointIndex1,1} ii]};
    mesh.vertexNtriangles(pointIndex2,1)={[mesh.vertexNtriangles{pointIndex2,1} ii]};
    mesh.vertexNtriangles(pointIndex3,1)={[mesh.vertexNtriangles{pointIndex3,1} ii]};    
end

for ii = 1:noOfpoints %find normals of all points
    polys = mesh.vertexNtriangles{ii};% get neighboring polygons to this point
    normal2 = zeros(1,3);
        
    for jj = 1 : size(polys,1)
        normal2 = normal2 + mesh.triangleNormals(polys(jj),:);
    end
    
    normal2 = normal2 / norm(normal2);
    mesh.vertexNormals(ii,:) = normal2;
end

for ii = 1 : noOfpolygons % find neighbouring polygons of all polygons
    polNeighbor = [];
    for jj = 1 : 3
        polNeighbor = [polNeighbor mesh.vertexNtriangles{mesh.triangles(ii,jj)}];
    end
    polNeighbor = unique(polNeighbor);
    polNeighbor = setdiff(polNeighbor, [ii]);
    mesh.triangleNtriangles(ii,1)={[polNeighbor]};
end

% --------------------------------------------------------------------
%                                                        Make PLY File
% -------------------------------------------------------------------- 
function makePly(mesh, fileName)

% makePly(mesh, fileName)
%
% this function converts a mesh to a *.ply file which can be rendered by 
% plyview from CyberWare and and Scanalyze from Stanford University
%
% mesh is a structure with the atleast the following two fields
% mesh.vertices and mesh.triangles
%
% Copyright : This code is written by Ajmal Saeed Mian {ajmal@csse.uwa.edu.au}
%              Computer Science, The University of Western Australia. The code
%              may be used, modified and distributed for research purposes with
%              acknowledgement of the author and inclusion this copyright information.
%
% Disclaimer : This code is provided as is without any warrantly.

noOfpoints = length(mesh.vertices);
noOfpolygons = length(mesh.triangles);

fid = fopen(fileName, 'wt');
fprintf(fid,'ply\nformat ascii 1.0\ncomment zipper output\nelement vertex %d\n', noOfpoints);
fprintf(fid,'property float x\nproperty float y\nproperty float z\nelement face %d\n', noOfpolygons);
fprintf(fid,'property list uchar int vertex_indices\nend_header\n');

for ii = 1 : noOfpoints
    fprintf(fid,'%f %f %f\n', mesh.vertices(ii,1), mesh.vertices(ii,2), mesh.vertices(ii,3));
end

polys = mesh.triangles;
polys = polys - 1;

for ii = 1 : noOfpolygons 
    fprintf(fid,'3 %d %d %d\n', polys(ii,1), polys(ii,2), polys(ii,3));
end
fclose(fid);