function model = load_model(dire,model_num)

model.dire=dire;
model.model_num=model_num;


dire=['data/' dire num2str(model_num)];

%parameters
model.bO=readNPY([dire '/bO.npy']);
model.bR=readNPY([dire '/bR.npy']);
model.wO=readNPY([dire '/wO.npy']);
model.wR=readNPY([dire '/wR.npy']);
model.wI=readNPY([dire '/wI.npy']);
model.h0=readNPY([dire '/h0.npy']);
%get fixed points
model.f1=readNPY([dire '/f1.npy']);
model.f2=readNPY([dire '/f2.npy']);
