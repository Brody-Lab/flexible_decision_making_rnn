function fixed_points = load_fixed_points(model)

dire=[model.dire num2str(model.model_num)];

%get fixed points
fixed_points.f1=readNPY(['data/' dire '/f1.npy']);
fixed_points.f2=readNPY(['data/' dire '/f2.npy']);

%output
fixed_points.o1=fixed_points.f1*model.wO'+model.bO;
fixed_points.o2=fixed_points.f2*model.wO'+model.bO;


%fixed point loss (stability)
new=tanh(model.wR*fixed_points.f1'+model.bR+model.wI(:,3));
mov1=new-fixed_points.f1';
fixed_points.loss1=sqrt(mean(mov1.^2));

new=tanh(model.wR*fixed_points.f2'+model.bR+model.wI(:,4));
mov2=new-fixed_points.f2';
fixed_points.loss2=sqrt(mean(mov2.^2));

