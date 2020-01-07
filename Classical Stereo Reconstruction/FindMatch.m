function [x1, x2] = FindMatch(I1, I2)

im1 = single(rgb2gray(I1));
im2 = single(rgb2gray(I2));

[f1,d1] = vl_sift(im1);
[f2,d2] = vl_sift(im2);

f1 = f1';
f2 = f2';
d1 = d1';
d2 = d2';

x1_left = [];
x2_left = [];
x1_right = [];
x2_right = [];
x1 = [];
x2 = [];

[idx, dist] = knnsearch(d2,d1,'K',2,'Distance','euclidean');

for i = 1:size(d1,1)
    if (dist(i,1)/dist(i,2) < 0.7)
        x1_left = [x1_left;f1(i,1:2)];
        x2_left = [x2_left;f2(idx(i,1),1:2)];
    end
end

[idx, dist] = knnsearch(d1,d2,'K',2,'Distance','euclidean');

for i = 1:size(d2,1)
    if (dist(i,1)/dist(i,2) < 0.7)
        x2_right = [x2_right;f2(i,1:2)];
        x1_right = [x1_right;f1(idx(i,1),1:2)];
    end
end

for i = 1:size(x1_left,1)
    for j = 1:size(x2_right,1)
        if( x2_left(i,1)==x2_right(j,1) && x2_left(i,2)==x2_right(j,2) && x1_right(j,1)==x1_left(i,1) && x1_right(j,2)==x1_left(i,2) )
            x1 = [x1;x1_left(i,:)];
            x2 = [x2;x2_left(i,:)];
        end
    end
end

[r1,c1,~] = size(I1);
[r2,c2,~] = size(I2);
combined_image = zeros(max(r1,r2),c1+c2,3,class(I1));
combined_image(1:r1,1:c1,:) = I1;
combined_image(1:r2,c1+1:c1+c2,:) = I2;

figure;
imshow(combined_image);
w = size(I1,2);
hold on;
n = size(x1,1);
for i = 1:n
    plot(x1(i,1),x1(i,2),'bo',x2(i,1)+w,x2(i,2),'bo');
    line([x1(i,1) x2(i,1)+w],[x1(i,2) x2(i,2)],'Color','green');
end