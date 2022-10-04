%%  Prediction model based on MOMSOA and MKELM
function [yPred,MAE,RMSE,MAPE]=MoMsoa_MKelm(p_train,t_train,p_val,t_val,p_test,t_test)
% Input Parameters:
% p_train    input data of training set
% t_train    output data of training set
% p_val      inout data of validation set
% t_val      output data of validation set
% p_test     input data of testing set
% t_test     output data of testing set
% Output Parameters:
% yPred          the prediction data
% mae        Mean absolute error
% rmse       root-mean-square error
% mape       Mean absolute percentage error

%% 数据归一化
[pn_train,ps] = mapminmax(p_train);%Normalize the input data
[tn_train,ts] = mapminmax(t_train);%Normalize the output data
pn_val = mapminmax('apply',p_val,ps);
pn_test = mapminmax('apply',p_test,ps);

%% 多目标参数优化 MoMsoa
Search_Agents = 10;%population size
Max_iterations =30;%maximum number of iterations
%Population coding rule：C(regularization coefficient)，σ^2（rbf kernel parameter），polynomial kernel parameter 1，polynomial kernel parameter 2，RBF kernel's weight coefficient（dimension=5）
Lower_bound = [0.01 0.01 0.01 1 0];
Upper_bound = [1 10 100 5 1];
dimension = 5;
fun_name = @MoMkelmFun; 
objective_no=2;%The number of optimization objective functions is 2
%Archive_F and Archive_X are Pareto front solutions and corresponding positions, respectively
[Archive_X,Archive_F]=MoMsoa(Search_Agents,Max_iterations,Lower_bound,Upper_bound,dimension,fun_name,objective_no,pn_train,tn_train,pn_val,t_val,ts);
%Accuracy and stability are equally important, so MAE and STD have the same weights of 0.5.  The first column of Archive_F represents MAE, and the second column of Archive_F represents STD
error=sum(Archive_F,2);%MAE+STD
[~,index]=sort(error);
BPosition=Archive_X(index(1),:);
%% 网络的建立和训练
C=BPosition(1); %Coefficient of regularization
Kernel_type='multi_kernel';
Kernel_para=BPosition(2:end);
y=elm_multiKernel(pn_train,tn_train,pn_test,C,Kernel_type,Kernel_para);
%The prediction results are inversely normalized
yPred=mapminmax('reverse',y,ts);
% error
MAE = mae(t_test,yPred);
RMSE = rmse(t_test,yPred);
MAPE = mape(t_test,yPred);
end