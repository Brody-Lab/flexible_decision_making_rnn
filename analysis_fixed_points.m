function fp_linsys = analysis_fixed_points(model,fixed_points)

unpackStruct(model);
unpackStruct(fixed_points);

n_units=length(wO);

%%% output at fixed points
fp_linsys.o1=f1*wO'+bO;
fp_linsys.o2=f2*wO'+bO;


% fp_linsys.sat1=nan(n_units,size(f1,1));
% fp_linsys.jacs1=nan(n_units,n_units,size(f1,1));
fp_linsys.in11=nan(n_units,size(f1,1));
fp_linsys.in21=nan(n_units,size(f1,1));
fp_linsys.r1=nan(n_units,size(f1,1));
fp_linsys.l1=nan(n_units,size(f1,1));
fp_linsys.dia1=nan(100,size(f2,1));

%context 1
disp('Loop through fixed points context 1')
for iii=1:size(f1,1)
    sat1=1-tanh(wR*f1(iii,:)'+bR+wI(:,3)).^2;
    jacs1=repmat(sat1,[1 n_units]).*wR;
    fp_linsys.in11(:,iii)=sat1.*wI(:,1);
    fp_linsys.in21(:,iii)=sat1.*wI(:,2);
    [V,D,W] = eig(jacs1);
    right_eigv=V;
    fp_linsys.dia1(:,iii)=diag(D(1:100,1:100));
    fp_linsys.r1(:,iii)=right_eigv(:,1)';
    left_eigv=inv(V);
    fp_linsys.l1(:,iii)=real(left_eigv(1,:))/norm(real(left_eigv(1,:)));    
end



fp_linsys.in12=nan(n_units,size(f2,1));
fp_linsys.in22=nan(n_units,size(f2,1));
fp_linsys.r2=nan(n_units,size(f2,1));
fp_linsys.l2=nan(n_units,size(f2,1));
fp_linsys.dia2=nan(100,size(f2,1));

%context 2
disp('Loop through fixed points context 2')
for iii=1:size(f2,1)    
    sat2=1-tanh(wR*f2(iii,:)'+bR+wI(:,4)).^2;   
    jacs2=repmat(sat2,[1 n_units]).*wR;
    fp_linsys.in12(:,iii)=sat2.*wI(:,1);
    fp_linsys.in22(:,iii)=sat2.*wI(:,2);    
    [V,D,W] = eig(jacs2);
    right_eigv=V;
    fp_linsys.dia2(:,iii)=diag(D(1:100,1:100));
    fp_linsys.r2(:,iii)=right_eigv(:,1)';
    left_eigv=inv(V);
    fp_linsys.l2(:,iii)=real(left_eigv(1,:))/norm(real(left_eigv(1,:)));  
end







