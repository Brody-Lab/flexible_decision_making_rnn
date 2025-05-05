function fixed_points = subsample_fixed_points(fixed_points,numbins,fixed_point_threshold)


unpackStruct(fixed_points);



bin_edges1=linspace(min(o1),max(o1)+eps,numbins+1);
ind1=[];
z=1;
for i=1:numbins
    indici=find(o1>=bin_edges1(i) & o1<bin_edges1(i+1));
    [val,ind1temp]=min(loss1(indici));
    if(val<fixed_point_threshold)
        ind1(z)=indici(ind1temp);
        z=z+1;
    end
end

bin_edges2=linspace(min(o2),max(o2)+eps,numbins+1);
ind2=[];
z=1;
for i=1:numbins
    indici=find(o2>=bin_edges2(i) & o2<bin_edges2(i+1));
    [val,ind2temp]=min(loss2(indici));
    if(val<fixed_point_threshold)
        ind2(z)=indici(ind2temp);
        z=z+1;
    end
end




fixed_points.f1=f1(ind1,:);
fixed_points.loss1=loss1(ind1);
fixed_points.o1=o1(ind1);

fixed_points.f2=f2(ind2,:);
fixed_points.loss2=loss2(ind2);
fixed_points.o2=o2(ind2);

