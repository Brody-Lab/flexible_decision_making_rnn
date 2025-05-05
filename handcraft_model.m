function model_handcrafted = handcraft_model(model,fixed_points,fp_linsys,components1,components2)


factor_weights = 1;

components1=abs(components1);
components1=components1/sum(components1);

components2=abs(components2);
components2=components2/sum(components2);

n_units=length(model.wO);

%%% output of the network computed at fixed points in context 1
output1=fp_linsys.o1;
%%% output of the network computed at fixed points in context 2
output2=fp_linsys.o2;


%%% I consider the linear system in each context around the fixed point
%%% associated with the smallest output (closest to the decision boundary).
%%% Similar results are obtained by choosing other fixed points
%%% as long as they are not towards the extremes.
%
%index of fixed point with smallest output in context 1
[~,ind1]=min(abs(output1));
%index of fixed point with smallest output in context 2
[~,ind2]=min(abs(output2));

% ind1=indice_fp;
% ind2=indice_fp;


fixed_point1=fixed_points.f1(ind1,:)';
fixed_point2=fixed_points.f2(ind2,:)';

%%% jacobian and effective index can be computed as the product
%%% of wR and wI respectively, and the derivative of the tanh
%%% around the fixed point
tanh_derivative1=1-tanh(model.wR*fixed_points.f1(ind1,:)'+model.bR+model.wI(:,3)).^2;
tanh_derivative2=1-tanh(model.wR*fixed_points.f2(ind2,:)'+model.bR+model.wI(:,4)).^2;



%jacobian for context 1
jacobian1=repmat(tanh_derivative1,[1 n_units]).*model.wR;
%jacobian for context 2
jacobian2=repmat(tanh_derivative2,[1 n_units]).*model.wR;





%first eigenvalue
disp(['Context 1, first eigenvalue :' num2str(abs(fp_linsys.dia1(1,ind1)))])



%right eigenvector (line attractor)
r1=fp_linsys.r1(:,ind1)';

%left eigenvector (selection vector)
%remove imaginary part due to precision error
l1=fp_linsys.l1(:,ind1)';




%%% assign sign of eigenvectors so that the line attractor
%%% points in the same direction as the RNN linear readout
if(corr(r1',model.wO')<0)
    r1=-r1;
    l1=-l1;
end

%%% angle between selection vector and line attractor
disp(['Context 1, angle between l and r = ' num2str(acosd(r1*l1'/norm(l1))) ])





%first eigenvalue
disp(['Context 2, first eigenvalue :' num2str(abs(fp_linsys.dia2(1,ind2)))])
%right eigenvector (line attractor)
r2=fp_linsys.r2(:,ind2)';
%left eigenvector (selection vector)
%remove imaginary part due to precision error
l2=fp_linsys.l2(:,ind2)';

%%% assign sign of eigenvectors so that the line attractor
%%% points in the same direction as the RNN linear readout
if(corr(r2',model.wO')<0)
    r2=-r2;
    l2=-l2;
end
%%% angle between selection vector and line attractor
disp(['Context 2, angle between l and r = ' num2str(acosd(r2*l2'/norm(l2))) ])






%%% average line attractor between the two contexts
average_r=(r1+r2)/2;

%component of selection vector orthogonal to line attractor
s1=l1-average_r;

%component of selection vector orthogonal to line attractor
s2=l2-average_r;


%%% average l between the two contexts
average_l=(l1+l2)/2;

%%% average s between the two contexts
average_s=(s1+s2)/2;


%%% l difference between the two contexts
difference_l=l2-l1;




%%% difference between tanh derivatives
difference_tanh_derivative=tanh_derivative1-tanh_derivative2;

%%% average tanh derivative
average_tanh_derivative=(tanh_derivative1+tanh_derivative2)/2;







model_handcrafted=model;

if(corr(average_r',model.wO')<0)
    average_r=-average_r;
end

model_handcrafted.wO=average_r;


model_handcrafted.h0=(fixed_point1+fixed_point2)/2;

model_handcrafted.h0s{1}=fixed_point1;
model_handcrafted.h0s{2}=fixed_point2;

input_factor=norm(model.wI(:,1));

clear cece
cece{1}=[tanh_derivative2.*l2' difference_tanh_derivative.*average_s' difference_tanh_derivative.* average_r' average_tanh_derivative.*difference_l' ]; %this all mech 1
cece{2}=[tanh_derivative2.*l2' average_tanh_derivative.*difference_l' difference_tanh_derivative.*average_s' difference_tanh_derivative.* average_r']; %this all mech 2
cece{3}=[tanh_derivative2.*l2' difference_tanh_derivative.* average_r' average_tanh_derivative.*difference_l' difference_tanh_derivative.*average_s']; %this all mech 3


new_input_weights_temp=nan(n_units,3);
for iii=1:3
    %find vector
    bibi=orth_mat(cece{iii});
    vec=bibi(:,end);
    %normalize
    vec=vec/norm(vec);
    %fix sign if necessary
    si=(tanh_derivative1.*vec)'*l1'-(tanh_derivative2.*vec)'*l2';
    if(si<0)
        vec=-vec;
    end
    %scale by desired weight
    new_input_weights_temp(:,iii)=components1(iii)*vec;
end
%take mean of three components
wnew=mean(new_input_weights_temp,2);
%normalize
wnew=wnew/norm(wnew);
%multiply by input factor
wnew=wnew*input_factor;

model_handcrafted.wI(:,1)=wnew*factor_weights;




%same for weights of frequency evidence (scale by components2)


clear cece
cece{1}=[tanh_derivative1.*l1' difference_tanh_derivative.*average_s' difference_tanh_derivative.* average_r' average_tanh_derivative.*difference_l' ]; %this all mech 1
cece{2}=[tanh_derivative1.*l1' average_tanh_derivative.*difference_l' difference_tanh_derivative.*average_s' difference_tanh_derivative.* average_r']; %this all mech 2
cece{3}=[tanh_derivative1.*l1' difference_tanh_derivative.* average_r' average_tanh_derivative.*difference_l' difference_tanh_derivative.*average_s']; %this all mech 3

new_input_weights_temp=nan(n_units,3);
for iii=1:3
    bibi=orth_mat(cece{iii});
    vec=bibi(:,end);
    vec=vec/norm(vec);
    si=(tanh_derivative2.*vec)'*l2'-(tanh_derivative1.*vec)'*l1';
    if(si<0)
        vec=-vec;
    end
    new_input_weights_temp(:,iii)=components2(iii)*vec;
end
wnew=mean(new_input_weights_temp,2);
wnew=wnew/norm(wnew);
wnew=wnew*input_factor;

model_handcrafted.wI(:,2)=wnew*factor_weights;



%%% I focus only on how the RNN differentially integrate the first input

%effective input for context 1
input1=tanh_derivative1'.*model_handcrafted.wI(:,1)';
%effective input for context 2
input2=tanh_derivative2'.*model_handcrafted.wI(:,1)';


%%% average input between the two contexts
average_input=(input1+input2)/2;


%%% average input between the two contexts
difference_input=input2-input1;




%%% final response difference
final_response_difference=l2*input2'-l1*input1';


% %%% rewriting of final response difference
final_response_difference2=average_input*difference_l'+average_l*difference_input';
% disp(['Final response difference, rewritten: ' num2str(final_response_difference2)]);

%check rewritten equation gives same result
if(abs(final_response_difference-final_response_difference2)>0.00001)
    error('?')
end


%%% splitting of final response difference into three terms
mechanism1=average_input*difference_l';
% disp(['Value mechanism 1: ' num2str(mechanism1)]);
mechanism2=average_r*difference_input';
% disp(['Value mechanism 2: ' num2str(mechanism2)]);
mechanism3=average_s*difference_input';
% disp(['Value mechanism 3: ' num2str(mechanism3)]);
% disp('****')
% disp('****')
% disp('****')
% disp('****')
% 



    

model_handcrafted.internals.fixed_point1=fixed_point1;
model_handcrafted.internals.fixed_point2=fixed_point2;
model_handcrafted.internals.tanh_derivative1=tanh_derivative1;
model_handcrafted.internals.tanh_derivative2=tanh_derivative2;
model_handcrafted.internals.jacobian1=jacobian1;
model_handcrafted.internals.jacobian2=jacobian2;
model_handcrafted.internals.l1=l1;
model_handcrafted.internals.l2=l2;
model_handcrafted.internals.r1=r1;
model_handcrafted.internals.r2=r2;
model_handcrafted.internals.s1=s1;
model_handcrafted.internals.s2=s2;
model_handcrafted.internals.average_r=average_r;
model_handcrafted.internals.average_l=average_l;
model_handcrafted.internals.average_s=average_s;
model_handcrafted.internals.average_tanh_derivative=average_tanh_derivative;
model_handcrafted.internals.difference_l=difference_l;
model_handcrafted.internals.difference_tanh_derivative=difference_tanh_derivative;
model_handcrafted.internals.input1=input1;
model_handcrafted.internals.input2=input2;
model_handcrafted.internals.average_input=average_input;
model_handcrafted.internals.difference_input=difference_input;
model_handcrafted.internals.final_response_difference=final_response_difference;
model_handcrafted.internals.mechanism1=mechanism1;
model_handcrafted.internals.mechanism2=mechanism2;
model_handcrafted.internals.mechanism3=mechanism3;

model_handcrafted.factor_weights=factor_weights;
    

   
