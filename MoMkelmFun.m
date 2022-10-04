function error=MoMkelmFun(pn_train, tn_train,pn_test,t_test,ts, C, Kernel_type, Kernel_para)
%t-test is the raw data, pn-train,tn-train,pn-test are the normalized
%data,ts is normalization rules for output data
y = elm_multiKernel(pn_train, tn_train,pn_test,C, Kernel_type,Kernel_para);
yPred = mapminmax('reverse',y,ts);
error(1)=mae(t_test,yPred);%mae
error(2)=std(t_test-yPred);%std
end