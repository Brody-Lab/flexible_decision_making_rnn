function [] = plot_2dresp(model)

unpackStruct(model.internals);


%number of time steps
maxt=40;

facto=0.9; %model_tau = 100ms


%%% pulse response in location context (wI(:,3))
clear r

fixed_point1=atanh(fixed_point1');

r(:,1)=fixed_point1;
for i=2:maxt
    if(i==2)
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,1)+model.wI(:,3));
    else
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,3));
    end
end
resp1=tanh(r);

resp1=resp1-repmat(tanh(fixed_point1)',[1 maxt]);



%%% pulse response in frequency context (wI(:,4))
clear r

fixed_point2=atanh(fixed_point2');

r(:,1)=fixed_point2;
for i=2:maxt
    if(i==2)
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,1)+model.wI(:,4));
    else
        r(:,i)=facto*r(:,i-1)+(1-facto)*(model.wR*tanh(r(:,i-1))+model.bR+model.wI(:,4));
    end
end
resp2=tanh(r);

resp2=resp2-repmat(tanh(fixed_point2)',[1 maxt]);


%%% orthogonalize average line attractor and average input
c=orth_mat([average_r' average_input']);


%%% plot response onto 2d space

hold on
plot(resp1'*c(:,1),resp1'*c(:,2),'.-','Color',[0 170 0]/255,'MarkerSize',10)
hold on
plot(resp2'*c(:,1),resp2'*c(:,2),'.-','Color',[255 0 0]/255,'MarkerSize',10)

hold on
plot(resp1(:,1)'*c(:,1),resp1(:,1)'*c(:,2),'.k','MarkerSize',30)
hold on
plot(resp2(:,1)'*c(:,1),resp2(:,1)'*c(:,2),'.k','MarkerSize',30)

xlabel('Projection on average r');
ylabel('Projection on average input');



