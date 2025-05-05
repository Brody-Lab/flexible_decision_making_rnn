function [] = plot_model_psychometrics(performance_matrix)

hold on

%average over location strengths
y3=mean(performance_matrix,1);
vec=[-4 -2.5 -1 1 2.5 4];

%%% compute logistic fit
ft = fittype( 'bao+t./(1+exp(-(x-alpha)/beta))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( ft );
opts.Display = 'Off';
opts.Lower = [0 0 0 0];
opts.StartPoint = [0.5 0.5 1 1];
opts.Upper = [0.1 2 15 15];
[fitresult, gof] = fit( vec', y3', ft, opts );
x=-4:.01:4;
y=fitresult.bao+fitresult.t./(1+exp(-(x-fitresult.alpha)/fitresult.beta));

plot(x,y,'b')
ylim([-.05 1.05])
xlim([-4 4])
box off
set(gca,'TickDir','out')
xlabel('evidence strength')
ylabel('% right choices')




hold on

%average over frequency strengths
y3=mean(performance_matrix,2)'; 
vec=[-4 -2.5 -1 1 2.5 4];

%%% compute logistic fit
ft = fittype( 'bao+t./(1+exp(-(x-alpha)/beta))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( ft );
opts.Display = 'Off';
opts.Lower = [0 0 0 0];
opts.StartPoint = [0.5 0.5 1 1];
opts.Upper = [0.1 2 15 15];
[fitresult, gof] = fit( vec', y3', ft, opts );
x=-4:.01:4;
y=fitresult.bao+fitresult.t./(1+exp(-(x-fitresult.alpha)/fitresult.beta));

plot(x,y,'r')
ylim([-.05 1.05])
xlim([-4 4])
box off
set(gca,'TickDir','out')
xlabel('Evidence strength')
ylabel('% right choices')


