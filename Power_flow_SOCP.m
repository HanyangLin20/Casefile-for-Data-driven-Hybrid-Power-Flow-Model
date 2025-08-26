 clc
 clear
 tic;
%% ��������
% mpc=IEEE33shunt_mo;
 % mpc=IEEE33re;
 mpc=case141shunt;

% mpc=grid_IEEE123;
T=1; 
% ��׼ֵ
Ub=mpc.bus(1,10);       % baseKV
Sb=mpc.baseMVA;         % baseMVA
Zb=Ub^2/Sb;             % base�迹��
Ib=Ub/Zb;               % baseKA
Yb=1/Zb;                % base����
% ���ۻ� ��mpc.branchֱ���ù�һ�������·����ʱ�����¾Ͳ��ù�һ����
r_pu=mpc.branch(:,3);     %��·����ı���ֵ
x_pu=mpc.branch(:,4);     %��·�翹�ı���ֵ
b_pu=mpc.branch(:,5);     %��·���ɵı���ֵ
Gs_pu=mpc.bus(:,5);       %�ڵ㲢���絼�ı���ֵ
Bs_pu=mpc.bus(:,6);       %�ڵ㲢�����ɵı���ֵ


%% �������
% �ڵ�֧·��
n_bus=size(mpc.bus,1);          %�ڵ�����
n_bra=size(mpc.branch,1);       %֧·����
n_gen=size(mpc.gen,1);         %���������

% �������(��������޹��ʣ���·���ʣ�ָ��fbus to tbus��,�ڵ��ѹfbus)
xp_gen=sdpvar(n_gen,T);         % �ڵ�ע���й����ʣ�ʵ��
xq_gen=sdpvar(n_gen,T);         % �ڵ�ע���޹����ʣ�ʵ��

xp_fbus=sdpvar(n_bra,1);        % ��·�й����ʣ�ָ��fbus to tbus��
xq_fbus=sdpvar(n_bra,1);        % ��·�޹����ʣ�ָ��fbus to tbus��

xv_bus=sdpvar(n_bus,1);          % �ڵ��ѹ��ƽ��
xtheta_bus=sdpvar(n_bus,1);      % �ڵ��ѹ�ĽǶ�
xh_bra=sdpvar(n_bra,T);          % ֧·������ƽ��

%% Ŀ�꺯��
f=sum(r_pu.*xh_bra);
% f=sum(sum(repmat(r_pu,1,T).*xh_bra));


%% Լ������ F=[F;...];
F=[];
% ����ڵ�*֧·��������ָ����С�ڵ����������ڵ�ţ�����Ϊ+1������Ϊ-1        
Abus_bra=zeros(n_bus,n_bra);
for i=1:n_bra  %����д��������
    Abus_bra(mpc.branch(i,1),i)=1;
    Abus_bra(mpc.branch(i,2),i)=-1;
end

% ����ڵ����Դ��Ĺ������󣬴��ڵ�Դ��
Agen=zeros(n_bus,n_gen);
for i=1:n_gen
    Agen(mpc.gen(i,1),i)=1;
end
% �����ڵ������빦�ʵ�����Ĺ�������
 Abus_bra2=Abus_bra;
 Abus_bra2(Abus_bra2==1)=0; % ȷ���������Ľڵ㣨����-1��������ڵ㣩
 Abus_bra2=-Abus_bra2;      % ȷ������Ӧ��Ϊ�������ʣ�Ӧ��Ϊ��
 
%% ���ڵ���д �й�ƽ�ⷽ��tempF1=����-���룬���涨����Ϊ��
% for t=1:T
% %      F=[F;Abus_bra*xp_fbus(:,t)+mpc.pload(:,t)/Sb+Gs_pu.*xv_bus(:,t)-Agen*xp_gen(:,t)-At*(r_pu.*xh_bra(:,t))==zeros(n_bus,1);];   %�ڵ�֧·���Ҫ��Ӧ�ϣ���������Ҫ��Ӧ��
%     F=[F;Abus_bra*xp_fbus(:,t)+mpc.pload(:,t)/Sb+Gs_pu.*xv_bus(:,t)-Agen*xp_gen(:,t)-At*(r_pu.*xh_bra(:,t))+Aess*xP_ess(:,t)/Sb==zeros(n_bus,1);];   %�ڵ�֧·���Ҫ��Ӧ�ϣ���������Ҫ��Ӧ��
% end
F=[F;Abus_bra*xp_fbus+mpc.bus(:,3)/Sb-Agen*xp_gen+Abus_bra2*(r_pu.*xh_bra)==zeros(n_bus,1);];   %�ڵ�֧·���Ҫ��Ӧ�ϣ���������Ҫ��Ӧ��

%% ���ڵ���д �޹�ƽ�ⷽ��tempF2=����-���룬���涨����Ϊ��
F=[F;Abus_bra*xq_fbus+mpc.bus(:,4)/Sb-Agen*xq_gen+Abus_bra2*(x_pu.*xh_bra)==zeros(n_bus,1);];

%% ��֧·��fbus to tbus��д��ѹ��Լ��
F=[F;Abus_bra'*xv_bus-2*(r_pu.*xp_fbus+x_pu.*xq_fbus)+(r_pu.^2+x_pu.^2).*xh_bra==zeros(n_bra,1);];

%% ��ѹ��ǵ�Լ�����ڷ���״�����п��Բ�д,�����ʽ��ϵ�ǽ��Ƶ�
F=[F;Abus_bra'*xtheta_bus-(x_pu.*xp_fbus-r_pu.*xq_fbus)/(1*1)==zeros(n_bra,1);];

%% ���䷽�̣����ڸ���������ʽԼ���ķ��̣�����׶�ɳ�
% %% ���䷽�̣����ڸ���������ʽԼ���ķ��̣�����׶�ɳ�
% for t=1:T
%     for i=1:n_bra
%         F=[F;rcone([xp_fbus(i,t);xq_fbus(i,t)],0.5*xh_bra(i,t),xv_bus(mpc.branch(i,1),t))];  % ����׶�ɳ�
%         %           F=[F;xp_fbus(i,t)^2+xq_fbus(i,t)^2==xh_bra(i,t)*xv_bus(mpc.branch(i,1),t)];      % ֱ��д��ʽԼ��
%     end
% end

for i=1:n_bra
    F=[F;rcone([xp_fbus(i,1);xq_fbus(i,1)],0.5*xh_bra(i,1),xv_bus(mpc.branch(i,1),1))];  % �����Ƕ���׶��ʽ
%F=[F;xp_fbus(i,1)^2+xq_fbus(i,1)^2<=xh_bra(i,1)*xv_bus(mpc.branch(i,1),1)];
end

%% ƽ��ڵ��ѹԼ���ͽڵ��ѹ�����ޱ߽�Լ��
index_1 = find(mpc.bus(:,2)==3);
index_2 = find(mpc.bus(:,2)~=3);
F = [F;xv_bus(index_1,1)==1.1025,0.9^2<=xv_bus(index_2,1)<=1.05^2];
% F = [F;xv_bus(index_1,1)==1.1025];      %ƽ��ڵ��ѹ��ֵ
% F = [F;xtheta_bus(index_1,1)==0];  %ƽ��ڵ��ѹ���

%% ���
% ops = sdpsettings('verbose',2,'solver', 'ipopt','savesolveroutput',1);
ops = sdpsettings('verbose',2,'solver', 'gurobi','savesolveroutput',1);
% ops = sdpsettings('verbose',2,'MaxIterations','10000','solver', 'mosek');
% ops = sdpsettings('verbose',2,'solver', 'scip');
% ops = sdpsettings('verbose',2,'solver', 'BONMIN');
result = solvesdp(F,f,ops);

%% ������
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
        Current(i,t) = xh_bra(i,t)*Ib^2;%��·������ƽ��
        Voltage(mpc.branch(i,1),t) = xv_bus(mpc.branch(i,1),t)*Ub^2;%��·��ѹ
        Pfbus(i,t) = xp_fbus(i,t)*Sb;%��·�й�����
        Qfbus(i,t) = xq_fbus(i,t)*Sb;%��·�޹�����
        error_socp(i,t) = Current(i,t)*Voltage(mpc.branch(i,1),t)-Pfbus(i,t)^2-Qfbus(i,t)^2;%��֧·����׶�滮�����
    end
end

Max_error_socp = max(abs(error_socp));%������׶���

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