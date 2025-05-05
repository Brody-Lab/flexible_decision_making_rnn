function data = create_stim(R, gamma_dir, gamma_freq, T, res, ntrials)


% rates of Poisson events on left and right
rrate = R/(exp(-gamma_dir)+1);
lrate = R - rrate;



% rates of Poisson events on left and right
hirate = R/(exp(-gamma_freq)+1);
lorate = R - hirate;



frac_hi=hirate./(hirate+lorate);
frac_lo=1-frac_hi;


rhirate=rrate*frac_hi;
rlorate=rrate*frac_lo;


lhirate=lrate*frac_hi;
llorate=lrate*frac_lo;

data.right_hi=poissrnd(rhirate*res,ntrials,T/res);
data.right_lo=poissrnd(rlorate*res,ntrials,T/res);
data.left_hi=poissrnd(lhirate*res,ntrials,T/res);
data.left_lo=poissrnd(llorate*res,ntrials,T/res);

