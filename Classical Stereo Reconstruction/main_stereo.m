
im1 = imread('left.bmp');
im2 = imread('right.bmp');

K = [350 0 960/2;
     0 350 538/2;
     0 0 1];
[x1, x2] = FindMatch(im1, im2);
[F] = ComputeF(x1, x2);

% Compute four configurations of camera pose given F
[R1, C1, R2, C2, R3, C3, R4, C4] = ComputeCameraPose(F, K);

%% Fill your code here
% Triangulate Points using four configurations
% e.g., P1: reference camera projection matrix at origin, P2: relative
% camera projection matrix with respect to P1
% X1 = Triangulation(P1, P2, x1, x2);

P0 = K*[eye(3),zeros(3,1)];
P1 = K*[R1,-R1*C1];
P2 = K*[R2,-R2*C2];
P3 = K*[R3,-R3*C3];
P4 = K*[R4,-R4*C4];

X1 = Triangulation(P0, P1, x1, x2);
X2 = Triangulation(P0, P2, x1, x2);
X3 = Triangulation(P0, P3, x1, x2);
X4 = Triangulation(P0, P4, x1, x2);

%%
% Disambiguate camera pose
[R,C,X] = DisambiguatePose(R1,C1,X1,R2,C2,X2,R3,C3,X3,R4,C4,X4);

% Stereo rectification
[H1, H2] = ComputeRectification(K, R, C);
im1_w = WarpImage(im1, H1);
im2_w = WarpImage(im2, H2);

figure
imshow(im1_w);
figure
imshow(im2_w);

im1_w = imresize(im1_w, 0.5);
im2_w = imresize(im2_w, 0.5);
[disparity] = DenseMatch(im1_w, im2_w);

figure
clf;
imagesc(disparity);
axis equal
axis off
colormap(jet);
