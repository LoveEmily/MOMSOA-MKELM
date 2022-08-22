clc;
clear;
close all;

% Change these details with respect to your problem%%%%%%%%%%%%%%
ObjectiveFunction=@ZDT1;
dim=5;
lb=0;
ub=1;
obj_no=2;
max_iter=100;
N=200;
[Archive_X,Archive_F]=MoMsoa(N,max_iter,lb,ub,dim,ObjectiveFunction,obj_no);

figure

Draw_ZDT1();

hold on

plot(Archive_F(:,1),Archive_F(:,2),'ro','MarkerSize',8,'markerfacecolor','k');

legend('True PF','Obtained PF');
title('MSSA');

set(gcf, 'pos', [403   466   230   200])