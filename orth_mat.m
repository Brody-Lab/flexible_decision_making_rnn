function mat = orth_mat(mat)

%orthogonalize all vectors in matrix mat using gram schmidt


c=mat;
%%% PROCEDURE TO MAKE THE VECTOR ORTHOGONAL
u=[];
for i=1:size(c,2)
    v=c(:,i);
    w=v;
    for j=1:i-1
        w=w-dot(v,u(:,j))*u(:,j);
    end
    u(:,i)=w/norm(w);
end
mat=u;
