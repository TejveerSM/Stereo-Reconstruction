function [H1, H2] = ComputeRectification(K, R, C)

rx = C/norm(C);
z = [0;0;1] - dot([0;0;1],rx)*rx;
rz = z/norm(z);
ry = cross(rz,rx);

R_rect = [rx';ry';rz'];

H1 = K*R_rect*inv(K);
H2 = K*R_rect*R'*inv(K);