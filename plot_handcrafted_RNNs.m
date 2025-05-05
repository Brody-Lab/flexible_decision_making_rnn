clear
clc
close all

dire='model_100_';


%%% seed of the base RNN
model_num=12 


compos=[0 logspace(-1,3,50)];
compos=compos(2:6:26);
compos=fliplr(compos)



%%% VARY WEIGHT OF D.I.M.
%%% (VERTICAL AXIS IN EXT.DATA FIG. 6)

f=figure;
set(f,'Position',[486 348 1129 636]);


for iii=1:length(compos)
    
    [iii length(compos)]
    
    model = load_model(dire,model_num);
    fixed_points = load_fixed_points(model);
    numbins=500;
    fixed_point_threshold=0.01;
    fixed_points = subsample_fixed_points(fixed_points,numbins,fixed_point_threshold);
    fp_linsys = analysis_fixed_points(model,fixed_points);
    
    % the three mechanism components are:
    % 1:s.v.m.; 2: d.i.m. ; 3: i.i.m;
    % here we are varying the strength of d.i.m. (second element)

    components1=[1 compos(iii) 1];
    components2=[1 compos(iii) 1];
    

    model_handcrafted = handcraft_model(model,fixed_points,fp_linsys,components1,components2);
    
    fp_linsys_handcrafted = analysis_fixed_points(model_handcrafted,fixed_points);
    
    [components1,components2] = localize_model(model_handcrafted,fixed_points,fp_linsys,0);
    
    
    
    svm_comp=components1(1);
    svm_comp=round(svm_comp*100)/100;
    dim_comp=components1(2);
    dim_comp=round(dim_comp*100)/100;
    iim_comp=components1(3);
    iim_comp=round(iim_comp*100)/100;



    subplot(4,5,iii);
    plot_2dresp(model_handcrafted)
    title(['D.I.M. = ' num2str(dim_comp)])
    % title(['SVM: ' num2str(svm_comp) ',DIM:' num2str(dim_comp) ',IIM:' num2str(iim_comp)])


    subplot(4,5,length(compos)+iii);
    plot_pulse_response(model_handcrafted,fixed_points,fp_linsys_handcrafted);
    title('Pulse response')


    subplot(4,5,length(compos)*2+iii);
    [performance_matrix_loc,performance_matrix_freq] = compute_model_psychometrics(model_handcrafted);
    plot_model_psychometrics(performance_matrix_loc)
    title('Psychometrics LOC context')
    

    subplot(4,5,length(compos)*3+iii);
    plot_model_psychometrics(performance_matrix_freq)
    title('Psychometrics FRQ context')
    


    
end







%%% KEEP D.I.M. = 0; VARY RELATIVE WEIGHT OF I.I.M. AND S.V.M.
%%% (HORIZONTAL AXIS IN EXT.DATA FIG. 6)

f2=figure;
set(f2,'Position',[486 348 1129 636]);


for iii=1:length(compos)
    
    [iii length(compos)]
    
    model = load_model(dire,model_num);
    fixed_points = load_fixed_points(model);
    numbins=500;
    fixed_point_threshold=0.01;
    fixed_points = subsample_fixed_points(fixed_points,numbins,fixed_point_threshold);
    fp_linsys = analysis_fixed_points(model,fixed_points);
    
    % the three mechanism components are:
    % 1:s.v.m.; 2: d.i.m. ; 3: i.i.m;
    % here we are varying the relative strength of s.v.m. (element 1) and i.i.m (element 3)

    components1=[compos(iii) 0 1-compos(iii)];
    components2=[compos(iii) 0 1-compos(iii)];
    

    model_handcrafted = handcraft_model(model,fixed_points,fp_linsys,components1,components2);
    
    fp_linsys_handcrafted = analysis_fixed_points(model_handcrafted,fixed_points);
    
    [components1,components2] = localize_model(model_handcrafted,fixed_points,fp_linsys,0);
    
    
    
    svm_comp=components1(1);
    svm_comp=round(svm_comp*100)/100;
    dim_comp=components1(2);
    dim_comp=round(dim_comp*100)/100;
    iim_comp=components1(3);
    iim_comp=round(iim_comp*100)/100;



    subplot(4,5,iii);
    plot_2dresp(model_handcrafted)
    title(['SVM: ' num2str(svm_comp) ',IIM:' num2str(iim_comp)])
    % title(['SVM: ' num2str(svm_comp) ',DIM:' num2str(dim_comp) ',IIM:' num2str(iim_comp)])


    subplot(4,5,length(compos)+iii);
    plot_pulse_response(model_handcrafted,fixed_points,fp_linsys_handcrafted);
    title('Pulse response')


    subplot(4,5,length(compos)*2+iii);
    [performance_matrix_loc,performance_matrix_freq] = compute_model_psychometrics(model_handcrafted);
    plot_model_psychometrics(performance_matrix_loc)
    title('Psychometrics LOC context')
    

    subplot(4,5,length(compos)*3+iii);
    plot_model_psychometrics(performance_matrix_freq)
    title('Psychometrics FRQ context')
    


    
end




