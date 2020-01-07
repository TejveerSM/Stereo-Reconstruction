function [R,C,X] = DisambiguatePose(R1,C1,X1,R2,C2,X2,R3,C3,X3,R4,C4,X4)

nValid = zeros(4,1);

np = R1*(X1'-C1*ones(1,size(X1,1)));
nValid(1,1) = length(find(np(3,:)>0 & X1(:,3)'>0));
np = R2*(X2'-C2*ones(1,size(X2,1)));
nValid(2,1) = length(find(np(3,:)>0 & X2(:,3)'>0));
np = R3*(X3'-C3*ones(1,size(X3,1)));
nValid(3,1) = length(find(np(3,:)>0 & X3(:,3)'>0));
np = R4*(X4'-C4*ones(1,size(X4,1)));
nValid(4,1) = length(find(np(3,:)>0 & X4(:,3)'>0));

[~,idx] = max(nValid);

if (idx == 1)
    R = R1;
    C = C1;
    X = X1;
elseif (idx == 2)
    R = R2;
    C = C2;
    X = X2;
elseif (idx == 3)
    R = R3;
    C = C3;
    X = X3;
else
    R = R4;
    C = C4;
    X = X4;
end