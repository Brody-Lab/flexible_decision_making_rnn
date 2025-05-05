function [components1,components2] = localize_model(model,fixed_points,fp_linsys,verbose)


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



%%% jacobian and effective index can be computed as the product
%%% of wR and wI respectively, and the derivative of the tanh
%%% around the fixed point
tanh_derivative1=1-tanh(model.wR*fixed_points.f1(ind1,:)'+model.bR+model.wI(:,3)).^2;
tanh_derivative2=1-tanh(model.wR*fixed_points.f2(ind2,:)'+model.bR+model.wI(:,4)).^2;



%jacobian for context 1
jacobian1=repmat(tanh_derivative1,[1 n_units]).*model.wR;
%jacobian for context 2
jacobian2=repmat(tanh_derivative2,[1 n_units]).*model.wR;





if(verbose)
    %first eigenvalue
    disp(['Context 1, first eigenvalue :' num2str(abs(fp_linsys.dia1(1,ind1)))])
end


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


if(verbose)
    %%% angle between selection vector and line attractor
    disp(['Context 1, angle between l and r = ' num2str(acosd(r1*l1'/norm(l1))) ])
end




%first eigenvalue

if(verbose)
    disp(['Context 2, first eigenvalue :' num2str(abs(fp_linsys.dia2(1,ind2)))])
end

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

if(verbose)
    %%% angle between selection vector and line attractor
    disp(['Context 2, angle between l and r = ' num2str(acosd(r2*l2'/norm(l2))) ])
end







% 
% %%% difference between tanh derivatives
% difference_tanh_derivative=tanh_derivative1-tanh_derivative2;
% 
% %%% average tanh derivative
% average_tanh_derivative=(tanh_derivative1+tanh_derivative2)/2;








%%% I focus only on how the RNN differentially integrate the first input

%effective input for context 1
input1=tanh_derivative1'.*model.wI(:,1)';
%effective input for context 2
input2=tanh_derivative2'.*model.wI(:,1)';




%%% average input between the two contexts
average_input=(input1+input2)/2;


%%% average input between the two contexts
difference_input=input1-input2;





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
difference_l=l1-l2;









%%% final response to pulse in context 1
final_response1=l1*input1';

%%% final response to pulse in context 2
final_response2=l2*input2';

%%% final response difference
final_response_difference=l1*input1'-l2*input2';

%%% rewriting of final response difference
final_response_difference2=average_input*difference_l'+average_l*difference_input';


%%% splitting of final response difference into three terms
mechanism1=average_input*difference_l';
mechanism2=average_r*difference_input';
mechanism3=average_s*difference_input';
components1=[mechanism1 mechanism2 mechanism3];
components1=abs(components1);
components1=components1/sum(components1);


if(verbose)
    
    disp(' ')
    disp(['Final response difference: ' num2str(final_response_difference)]);
    disp(['Final response difference, rewritten: ' num2str(final_response_difference2)]);
    
    disp(['Value mechanism 1: ' num2str(mechanism1)]);
    disp(['Value mechanism 2: ' num2str(mechanism2)]);
    disp(['Value mechanism 3: ' num2str(mechanism3)]);
    disp('****')
    disp('****')
    disp('****')
    disp('****')
    
    
    
    
    
    disp('  ');
    disp('  ');
    disp('  ');
    disp('  ');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('**************$$$$$$$$$$$$$$$$$****************');
    disp('  ');
    disp('  ');
    disp('  ');
    disp('  ');
    
    
    
end





%%% I focus only on how the RNN differentially integrate the first input

%effective input for context 1
input1=tanh_derivative1'.*model.wI(:,2)';
%effective input for context 2
input2=tanh_derivative2'.*model.wI(:,2)';


%%% average input between the two contexts
average_input=(input1+input2)/2;


%%% average input between the two contexts
difference_input=input2-input1;





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






%%% final response to pulse in context 1
final_response1=l1*input1';

%%% final response to pulse in context 2
final_response2=l2*input2';

%%% final response difference
final_response_difference=l2*input2'-l1*input1';

%%% rewriting of final response difference
final_response_difference2=average_input*difference_l'+average_l*difference_input';


%%% splitting of final response difference into three terms
mechanism1=average_input*difference_l';
mechanism2=average_r*difference_input';
mechanism3=average_s*difference_input';


components2=[mechanism1 mechanism2 mechanism3];
components2=abs(components2);
components2=components2/sum(components2);



if(verbose)
    disp(' ')
    disp(['Final response difference: ' num2str(final_response_difference)]);
    disp(['Final response difference, rewritten: ' num2str(final_response_difference2)]);
    disp(['Value mechanism 1: ' num2str(mechanism1)]);
    disp(['Value mechanism 2: ' num2str(mechanism2)]);
    disp(['Value mechanism 3: ' num2str(mechanism3)]);
    disp('****')
    disp('****')
    disp('****')
    disp('****')
end





