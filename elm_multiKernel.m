function TY = elm_multiKernel(TrainingData_input, TrainingData_output,TestingData_input, Regularization_coefficient, Kernel_type, Kernel_para)
% Input:
% TrainingData_File           - Filename of training data set
% TestingData_File            - Filename of testing data set
% Regularization_coefficient  - Regularization coefficient C
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
% Output: 
% Sample2 classification: elm_kernel('diabetes_train', 'diabetes_test', 1, 1, 'RBF_kernel',100)

%%%%%%%%%%% Load training dataset
T=TrainingData_output;
P=TrainingData_input;

%%%%%%%%%%% Load testing dataset
TV.P=TestingData_input;

C = Regularization_coefficient;

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(T,2);
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T')); 

%%%%%%%%%%% Calculate the training output
% Y=(Omega_train * OutputWeight)';                             %   Y: the actual output of the training data

%%%%%%%%%%% Calculate the output of testing input

Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data
    
    
%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'RBF_kernel')
    if nargin<4
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'multi_kernel')%kernel_pars contains the weighting coefficient of multiple kernels
    if nargin<4
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega1 = XXh+XXh'-2*(Xtrain*Xtrain');
        omega1 = exp(-omega1./kernel_pars(1));%RBF kernel
%         omega2 = Xtrain*Xtrain';
        omega2 = (Xtrain*Xtrain'+kernel_pars(2)).^kernel_pars(3);%polynomial kernel
        omega = kernel_pars(4).*omega1+(1-kernel_pars(4)).*omega2;
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega1 = XXh1+XXh2' - 2*Xtrain*Xt';
        omega1 = exp(-omega1./kernel_pars(1));%RBF kernel
%         omega2 = Xtrain*Xt';
       omega2 = (Xtrain*Xt'+kernel_pars(2)).^kernel_pars(3);
        omega = kernel_pars(4).*omega1+(1-kernel_pars(4)).*omega2;
    end
end

