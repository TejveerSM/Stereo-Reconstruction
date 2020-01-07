function [F] = ComputeF(x1, x2)

F = zeros(3,3);
inliers = 0;
A = zeros(8,9);
threshold = 0.01;
for k = 1:100000
    r = randperm(size(x1,1),8);
    local_inliers = 0;
    for i = 1:8
        A(i,1) = x1(r(i),1)*x2(r(i),1);
        A(i,2) = x1(r(i),2)*x2(r(i),1);
        A(i,3) = x2(r(i),1);
        A(i,4) = x1(r(i),1)*x2(r(i),2);
        A(i,5) = x1(r(i),2)*x2(r(i),2);
        A(i,6) = x2(r(i),2);
        A(i,7) = x1(r(i),1);
        A(i,8) = x1(r(i),2);
        A(i,9) = 1;
    end    
    [~,~,Va] = svd(A'*A);
    x = Va(:,end);
    F_local = reshape(x,3,3)';
    F_local = F_local/F_local(3,3);
    [u,d,v] = svd(F_local);
    d(3, 3) = 0;
    F_local = u*d*v';
    for i = 1:size(x1,1)
        u = [x1(i,1);x1(i,2);1];
        vt = [x2(i,1),x2(i,2),1];
        error = abs(vt*F_local*u);
        if(error<threshold)
            local_inliers = local_inliers + 1;
        end
    end
    if (k==1)
        F = F_local;
        inliers = local_inliers;
    elseif (local_inliers > inliers)
        F = F_local;
        inliers = local_inliers;
    end
end