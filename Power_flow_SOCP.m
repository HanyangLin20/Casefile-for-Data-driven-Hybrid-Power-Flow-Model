 clc
 clear
 tic;
%% 输入数据
% mpc=IEEE33shunt_mo;
 % mpc=IEEE33re;
 mpc=case141shunt;

% mpc=grid_IEEE123;
T=1; 
% 基准值
Ub=mpc.bus(1,10);       % baseKV
Sb=mpc.baseMVA;         % baseMVA
Zb=Ub^2/Sb;             % base阻抗Ω
Ib=Ub/Zb;               % baseKA
Yb=1/Zb;                % base导纳
% 标幺化 当mpc.branch直接用归一化后的线路数据时，以下就不用归一化了
r_pu=mpc.branch(:,3);     %线路电阻的标幺值
x_pu=mpc.branch(:,4);     %线路电抗的标幺值
b_pu=mpc.branch(:,5);     %线路电纳的标幺值
Gs_pu=mpc.bus(:,5);       %节点并联电导的标幺值
Bs_pu=mpc.bus(:,6);       %节点并联电纳的标幺值


%% 定义变量
% 节点支路数
n_bus=size(mpc.bus,1);          %节点数量
n_bra=size(mpc.branch,1);       %支路数量
n_gen=size(mpc.gen,1);         %发电机数量

% 定义变量(发电机有无功率，线路功率（指从fbus to tbus）,节点电压fbus)
xp_gen=sdpvar(n_gen,T);         % 节点注入有功功率，实型
xq_gen=sdpvar(n_gen,T);         % 节点注入无功功率，实型

xp_fbus=sdpvar(n_bra,1);        % 线路有功功率（指从fbus to tbus）
xq_fbus=sdpvar(n_bra,1);        % 线路无功功率（指从fbus to tbus）

xv_bus=sdpvar(n_bus,1);          % 节点电压的平方
xtheta_bus=sdpvar(n_bus,1);      % 节点电压的角度
xh_bra=sdpvar(n_bra,T);          % 支路电流的平方

%% 目标函数
f=sum(r_pu.*xh_bra);
% f=sum(sum(repmat(r_pu,1,T).*xh_bra));


%% 约束条件 F=[F;...];
F=[];
% 先求节点*支路关联矩阵，指定从小节点号流出到大节点号，流出为+1，流入为-1        
Abus_bra=zeros(n_bus,n_bra);
for i=1:n_bra  %逐列写关联矩阵
    Abus_bra(mpc.branch(i,1),i)=1;
    Abus_bra(mpc.branch(i,2),i)=-1;
end

% 再求节点与电源点的关联矩阵，存在电源点
Agen=zeros(n_bus,n_gen);
for i=1:n_gen
    Agen(mpc.gen(i,1),i)=1;
end
% 最后求节点与流入功率的网损的关联矩阵
 Abus_bra2=Abus_bra;
 Abus_bra2(Abus_bra2==1)=0; % 确定所关联的节点（保留-1，即流入节点）
 Abus_bra2=-Abus_bra2;      % 确定网损应该为流出功率，应该为正
 
%% 按节点列写 有功平衡方程tempF1=流出-流入，即规定流出为正
% for t=1:T
% %      F=[F;Abus_bra*xp_fbus(:,t)+mpc.pload(:,t)/Sb+Gs_pu.*xv_bus(:,t)-Agen*xp_gen(:,t)-At*(r_pu.*xh_bra(:,t))==zeros(n_bus,1);];   %节点支路编号要对应上，发电机编号要对应上
%     F=[F;Abus_bra*xp_fbus(:,t)+mpc.pload(:,t)/Sb+Gs_pu.*xv_bus(:,t)-Agen*xp_gen(:,t)-At*(r_pu.*xh_bra(:,t))+Aess*xP_ess(:,t)/Sb==zeros(n_bus,1);];   %节点支路编号要对应上，发电机编号要对应上
% end
F=[F;Abus_bra*xp_fbus+mpc.bus(:,3)/Sb-Agen*xp_gen+Abus_bra2*(r_pu.*xh_bra)==zeros(n_bus,1);];   %节点支路编号要对应上，发电机编号要对应上

%% 按节点列写 无功平衡方程tempF2=流出-流入，即规定流出为正
F=[F;Abus_bra*xq_fbus+mpc.bus(:,4)/Sb-Agen*xq_gen+Abus_bra2*(x_pu.*xh_bra)==zeros(n_bus,1);];

%% 按支路（fbus to tbus）写电压降约束
F=[F;Abus_bra'*xv_bus-2*(r_pu.*xp_fbus+x_pu.*xq_fbus)+(r_pu.^2+x_pu.^2).*xh_bra==zeros(n_bra,1);];

%% 电压相角的约束，在辐射状网络中可以不写,这个等式关系是近似的
F=[F;Abus_bra'*xtheta_bus-(x_pu.*xp_fbus-r_pu.*xq_fbus)/(1*1)==zeros(n_bra,1);];

%% 补充方程，关于辅助变量等式约束的方程，二阶锥松弛
% %% 补充方程，关于辅助变量等式约束的方程，二阶锥松弛
% for t=1:T
%     for i=1:n_bra
%         F=[F;rcone([xp_fbus(i,t);xq_fbus(i,t)],0.5*xh_bra(i,t),xv_bus(mpc.branch(i,1),t))];  % 二阶锥松弛
%         %           F=[F;xp_fbus(i,t)^2+xq_fbus(i,t)^2==xh_bra(i,t)*xv_bus(mpc.branch(i,1),t)];      % 直接写等式约束
%     end
% end

for i=1:n_bra
    F=[F;rcone([xp_fbus(i,1);xq_fbus(i,1)],0.5*xh_bra(i,1),xv_bus(mpc.branch(i,1),1))];  % 这里是二阶锥形式
%F=[F;xp_fbus(i,1)^2+xq_fbus(i,1)^2<=xh_bra(i,1)*xv_bus(mpc.branch(i,1),1)];
end

%% 平衡节点电压约束和节点电压上下限边界约束
index_1 = find(mpc.bus(:,2)==3);
index_2 = find(mpc.bus(:,2)~=3);
F = [F;xv_bus(index_1,1)==1.1025,0.9^2<=xv_bus(index_2,1)<=1.05^2];
% F = [F;xv_bus(index_1,1)==1.1025];      %平衡节点电压幅值
% F = [F;xtheta_bus(index_1,1)==0];  %平衡节点电压相角

%% 求解
% ops = sdpsettings('verbose',2,'solver', 'ipopt','savesolveroutput',1);
ops = sdpsettings('verbose',2,'solver', 'gurobi','savesolveroutput',1);
% ops = sdpsettings('verbose',2,'MaxIterations','10000','solver', 'mosek');
% ops = sdpsettings('verbose',2,'solver', 'scip');
% ops = sdpsettings('verbose',2,'solver', 'BONMIN');
result = solvesdp(F,f,ops);

%% 输出结果
xp_gen  = double(xp_gen);
xq_gen  = double(xq_gen);
xp_fbus = double(xp_fbus);
xq_fbus = double(xq_fbus);
xv_bus  = double(xv_bus);
xh_bra  = double(xh_bra);
xtheta_bus=double(xtheta_bus);
xtheta_bus=xtheta_bus*180/pi;
 
 
Vbus = sqrt(xv_bus);
Ibra = sqrt(xh_bra);

for t=1:T
    for i=1:n_bra
        Current(i,t) = xh_bra(i,t)*Ib^2;%线路电流的平方
        Voltage(mpc.branch(i,1),t) = xv_bus(mpc.branch(i,1),t)*Ub^2;%线路电压
        Pfbus(i,t) = xp_fbus(i,t)*Sb;%线路有功功率
        Qfbus(i,t) = xq_fbus(i,t)*Sb;%线路无功功率
        error_socp(i,t) = Current(i,t)*Voltage(mpc.branch(i,1),t)-Pfbus(i,t)^2-Qfbus(i,t)^2;%各支路二阶锥规划的误差
    end
end

Max_error_socp = max(abs(error_socp));%最大二阶锥误差

f = double(f)
Max_error_socp = double(Max_error_socp)
toc
% errorLBFV=Vbus-xv;
% errorLBFSV=Vbus-xV;
% abt=runpf('case69');
% % toc
% U_mag = value(Vbus);
% plot(Vbus,'b-+');
% ylabel('Voltage Magnitude');
% xlabel('Node Numer');

% 
% U_mag = value(xp_fbus);
% plot(Current,'b-+');
% ylabel('Power Magnitude');
% xlabel('Branch Numer');