function [perf_matrix_loc,perf_matrix_frq] = compute_model_psychometrics(model)


unpackStruct(model)

n_units=length(wO);

facto=0.9; %model_tau = 100ms


easiest=4; %evidence strength for easiest trial

ntrials=10; %number of trials per condition


gi=linspace(-easiest,easiest,6);

for i=1:length(gi)
    for j=1:length(gi)
        
        
        data = create_stim(40, gi(i), gi(j), 1.3, 0.01, ntrials);

        excess_right=(data.right_hi+data.right_lo)-(data.left_hi+data.left_lo);
        excess_high=(data.right_hi+data.left_hi)-(data.right_lo+data.left_lo);
        inputs=zeros(ntrials,130,4);
        inputs(:,:,1)=excess_right;
        inputs(:,:,2)=excess_high;
        inputs(:,:,3)=1; %%% set location context input to 1

        %%% predict responses to full trials
        hid=nan(ntrials,130,n_units);
        outz=nan(ntrials,130);
        for iii=1:ntrials
            in=squeeze(inputs(iii,:,:))';
            
            r=nan(n_units,130);
            r(:,1)=h0;
            for a=2:130
                r(:,a)=facto*r(:,a-1)+(1-facto)*(wR*tanh(r(:,a-1))+wI*in(:,a)+bR);
            end
            hid(iii,:,:)=tanh(r');
            outz(iii,:)=wO*tanh(r)+bO;
    
        end

        perf_matrix_loc(i,j)=mean(sign(outz(:,end)));
        
        
        

        
        
        inputs=zeros(ntrials,130,4);
        inputs(:,:,1)=excess_right;
        inputs(:,:,2)=excess_high;
        inputs(:,:,4)=1; %%% set frequency context input to 1
        
        %%% predict responses to full trials
        hid=nan(ntrials,130,n_units);
        outz=nan(ntrials,130);
        for iii=1:ntrials
            in=squeeze(inputs(iii,:,:))';
            
            r=nan(n_units,130);
            r(:,1)=h0;
            for a=2:130
                r(:,a)=facto*r(:,a-1)+(1-facto)*(wR*tanh(r(:,a-1))+wI*in(:,a)+bR);
            end
            hid(iii,:,:)=tanh(r');
            outz(iii,:)=wO*tanh(r)+bO;
     
        end

        perf_matrix_frq(i,j)=mean(sign(outz(:,end)));        
        
    end
end


%%% change range of responses from (-1,1) to (0,1)
perf_matrix_loc=(perf_matrix_loc+1)/2;
perf_matrix_frq=(perf_matrix_frq+1)/2;



end

