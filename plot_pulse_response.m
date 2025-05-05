function [resp11,resp12,c,slope_val,d_shap] = plot_pulse_response(model,fixed_points,fp_linsys)



n_units=length(model.wO);



facto=0.9; %model_tau = 100ms

% number of time points
maxt=50;

xtime=linspace(0,0.65,maxt);


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




%%% jacobian and effective index can be computed as the product
%%% of wR and wI respectively, and the derivative of the tanh
%%% around the fixed point
tanh_derivative1=1-tanh(model.wR*fixed_points.f1(ind1,:)'+model.bR+model.wI(:,3)).^2;
tanh_derivative2=1-tanh(model.wR*fixed_points.f2(ind2,:)'+model.bR+model.wI(:,4)).^2;


%jacobian for context 1
jacobian1=repmat(tanh_derivative1,[1  n_units]).*model.wR;
%jacobian for context 2
jacobian2=repmat(tanh_derivative2,[1  n_units]).*model.wR;





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


z=1;



%%% pulse response in location context (wI(:,3))

clear r

fixed_point1=atanh(fixed_points.f1(ind1,:));

r(:,1)=fixed_point1;
for i=2:maxt
    if(i==2)
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,1)+model.wI(:,3));
    else
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,3));
    end
end
resp11=tanh(r);

resp11=resp11-repmat(tanh(fixed_point1)',[1 maxt]);



%%% pulse response in frequency context (wI(:,4))
clear r

fixed_point2=atanh(fixed_points.f2(ind2,:));

r(:,1)=fixed_point2;
for i=2:maxt
    if(i==2)
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,1)+model.wI(:,4));
    else
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,4));
    end
end
resp12=tanh(r);

resp12=resp12-repmat(tanh(fixed_point2)',[1 maxt]);





c=average_r';


%%% compute differential response

resp1=resp11-resp12;
resp2=resp12-resp11;




hold on
plot(xtime,resp1'*c(:,1),'-','Color',[130 130 130]/255,'MarkerSize',10,'LineWidth',2)
hold on
plot(xtime,resp2'*c(:,1),'-','Color',[130 130 130]/255,'MarkerSize',10,'LineWidth',2)



hold on
plot(xtime(2:12:50),resp1(:,2:12:50)'*c(:,1),'^','Color',[130 130 130]/255,'MarkerSize',10,'MarkerFaceColor',[130 130 130]/255,'LineWidth',2)
hold on
plot(xtime(2:12:50),resp2(:,2:12:50)'*c(:,1),'^','Color',[130 130 130]/255,'MarkerSize',10,'MarkerFaceColor',[130 130 130]/255,'LineWidth',2)


xlabel('Time')
ylabel('Differential pulse response')
set(gca,'TickDir','out')
box off
axis tight





