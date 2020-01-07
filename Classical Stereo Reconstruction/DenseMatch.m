function [disparity] = DenseMatch(im1, im2)

I1 = rgb2gray(im1);
I2 = rgb2gray(im2);

disparity = zeros(size(I1));

I1 = [zeros(size(I1,1),6),I1,zeros(size(I1,1),6)];
I1 = [zeros(6,size(I1,2));I1;zeros(6,size(I1,2))];

I2 = [zeros(size(I2,1),6),I2,zeros(size(I2,1),6)];
I2 = [zeros(6,size(I2,2));I2;zeros(6,size(I2,2))];

[~,d1] = vl_dsift(single(I1),'size',4);
[~,d2] = vl_dsift(single(I2),'size',4);

d1_mat = reshape(d1',[size(im1,1),size(im1,2),128]);
d2_mat = reshape(d2',[size(im2,1),size(im2,2),128]);

for i = 1:size(im1,1)
    for j = 1:size(im1,2)
        v = 1;
        low_dist = sum(abs(d1_mat(i,j,:)-d2_mat(i,1,:)));
        for k = 1:j
            dist = sum(abs(d1_mat(i,j,:)-d2_mat(i,k,:)));
            if(dist<low_dist)
                low_dist = dist;
                v = k;
            end
        end
        disparity(i,j) = abs(j-v);
    end
end