function [X] = Triangulation(P1, P2, x1, x2)

rows = size(x1,1);
X = zeros(rows,3);

for i = 1:rows
    a1 = x1(i,:);
    s1 = [0,-1,a1(2);1,0,-a1(1);-a1(2),a1(1),0];
    a2 = x2(i,:);
    s2 = [0,-1,a2(2);1,0,-a2(1);-a2(2),a2(1),0];
    A = [s1*P1; s2*P2];
    [u,d,v] = svd(A);
    X(i,:) = (v(1:3,end)/v(end,end))';    
end