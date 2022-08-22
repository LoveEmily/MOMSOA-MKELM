
%%% Designed and Developed by Dr. Gaurav Dhiman (http://dhimangaurav.com/) %%%
% Improved multi-objective gull optimization algorithm. The standard SOA
% algorithm is designed by Dr. Gaurav Dhiman

function[Archive_X,Archive_F]=MoMsoa(Search_Agents,Max_iterations,Lower_bound,Upper_bound,dimension,objective_fun,objective_no,varargin)

if size(Upper_bound,2)==1
    Upper_bound=ones(1,dimension)*Upper_bound;
    Lower_bound=ones(1,dimension)*Lower_bound;
end
ArchiveMaxSize=100;% Archinve size, the max number of pareto solutions
Archive_X=zeros(100,dimension);% Pareto solutions
Archive_F=ones(100,objective_no)*inf;% The fitness value corresponding to the Pareto solutions
Archive_member_no=0;% The number of current Pareto optimal solutions

fitness=zeros(Search_Agents,objective_no);% Initialize the fitness value
Best_fitness=inf*ones(1,objective_no); % Initialize the optimal fitness value
Best_position=zeros(dimension,1); % Initialize the position of optimal search agent

% Using Tent map initialize the population positions to improve the initial population diversity
% Positions=init(Search_Agents,dimension,Upper_bound,Lower_bound);
z=rand(1,dimension);
z(1)=0.4835;
for i=1:Search_Agents
    for j=1:dimension
        
        if z(j)<=0.51 && z(j)>=0
            z(j)=2*z(j);
        elseif z(j)<=1 && z(j)>0.5
            z(j)=2*(1-z(j));
        end
    end
    Positions(i,:) = Lower_bound+z.*(Upper_bound-Lower_bound);
end

l=0;
while l<Max_iterations
    for i=1:size(Positions,1) 
        Positions(i,:)=boundtest(Positions(i,:),Lower_bound,Upper_bound); %bounds checking

        fitness(i,:)=objective_fun(Positions(i,:));
        if dominates(fitness(i,:),Best_fitness)
            Best_fitness=fitness(i,:);
            Best_position=Positions(i,:);
        end
    end

    [Archive_X, Archive_F, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, Positions, fitness, Archive_member_no);

    if Archive_member_no>ArchiveMaxSize
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, objective_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no]=HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, objective_no);
    else
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, objective_no);
    end
    index=RouletteWheelSelection(1./Archive_mem_ranks);
    if index==-1
        index=1;
    end
    Best_fitness=Archive_F(index,:);
    Best_position=Archive_X(index,:);
    
%     A=2-l*((2)/Max_iterations); %Fc£º2->0 linear decrease
    A=(100-200*(l/Max_iterations))./(1+abs(100-200*(l/Max_iterations)));
    u=(Lower_bound+(Upper_bound-Lower_bound)*cos((l*pi)./(Max_iterations*2)))*10^-2;
    v=rand();
    for i=1:size(Positions,1)
        r1=rand();
        k=rand()*2*pi;
        B=2*A^2*r1;
        D_alphs=A.*Positions(i,:)+B.*((Best_position-Positions(i,:)));
        
        r=u*exp(k*v);
        x=r*cos(k);
        y=r*sin(k);
        z=r*k;
        P=x.*y.*z;
        Positions(i,:)=D_alphs.*P+Best_position;
    end
    l=l+1;    
end
end
function newposition=boundtest(position,lb,ub)
Flag4Upper_bound=position>ub;
Flag4Lower_bound=position<lb;
newposition=(position.*(~(Flag4Upper_bound+Flag4Lower_bound)))+ub.*Flag4Upper_bound+lb.*Flag4Lower_bound;
end