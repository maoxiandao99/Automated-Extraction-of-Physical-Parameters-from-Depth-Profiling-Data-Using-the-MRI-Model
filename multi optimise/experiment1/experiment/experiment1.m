clear
clc
tic
%% 导入数据

inddis=0.5;

% % w = 2.2;
% % lambda = 1.2;
% % sigema_0=0.2;
% % sigema_k=0;
% % sigema_c=0;
original_w = 2.2;
original_lambda = 2.2;
original_sigema_0 = 0.9;
original_sigema_k = 0.019253260754461;
original_sigema_c = 0;
%构造膜层结构
obj_tn=[16 14.5 15 15 15 15 15 15];
obj_tn=[obj_tn 10.388];
% 
cons=10;


%------------------------------------------------------------------------------------------------------------------------
%% sigma
%线性
sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0+sigema_k*(i*inddis);
%常数
 %sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0;
%开根号
 %sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sqrt(sigema_0+sigema_k*(i*inddis)^2);
%指数
% sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_k*sigema_0^(sigema_c*(i*inddis));
%------------------------------------------------------------------------------------------------------------------------

% %  导入数据
Cr=xlsread("C:\Users\Xiangxiang\Desktop\多参改版\第一个实验数据\灵敏度分析\experimentdata1.xlsx");
ra2_ex = Cr(:,1);%深度
data = Cr(:,2);%信号强度
% data = data/100;%信号强度归一化
ra2_ex = ra2_ex';
ra2=ra2_ex;
data = data';
wide_ex=max(ra2_ex);
% % %------------------------------------------------------------------------------------------------------------------------
%去除奇异值
%  ra2_ex = ra2_ex(isfinite(ra2_ex));%去除奇异值
% data = data(isfinite(data));
% data=smooth(data,'rloess');%光滑之后可能会报错
% plot(ra2_ex,data);
%一维插值
x1_ex=ra2_ex;
y1_ex=data;
ra2_ex=0:inddis:max(ra2_ex);
data=interp1(x1_ex,y1_ex,ra2_ex,'spline');
data(data>1)=1;%简化版的判断语句
data(data<0)=0;
data=data';
fit=data;   %fit=拟合结果
plot(ra2_ex,data);
DataLength = length(fit);  % 记录输入的实验数据数组的大小


%% 定义参数范围
DefineDomain = [0 0 0 0 0;
                3 3 3 1 0]; % w, lambda, sigema_0, sigema_k, sigema_c

%% CPSO参数设置
numSubSwarms = 10; % 子群数量
popSize = 5; % 每个子群的粒子数量
maxIterations = 200; % 最大迭代次数
w_max = 0.9;     %a小优化，针对不同情况可以考虑，动态调整三个参数的数值，先通过侧重局部最优后侧重全局最优
w_min = 0.4;
c1_initial = 1.5;
c2_initial = 1.5;
D = 5; % 参数维度
subSwarmPositions = zeros(popSize, D, numSubSwarms);
subSwarmVelocities = zeros(popSize, D, numSubSwarms);
subSwarmBestPositions = zeros(popSize, D, numSubSwarms);
subSwarmBestFitness = Inf(popSize, numSubSwarms);

globalBestPosition = zeros(1, D);
globalBestFitness = Inf;
fitnessHistory = zeros(maxIterations, 1);

%% 初始化子群粒子位置和速度
for i = 1:numSubSwarms
    for j = 1:popSize
        subSwarmPositions(j, :, i) = DefineDomain(1,:) + rand(1, D) .* (DefineDomain(2,:) - DefineDomain(1,:));
        subSwarmVelocities(j, :, i) = zeros(1, D);
        
        % 计算初始适应度
        fitness = sqrt(mean((convolution1(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5)) - fit).^2));
        subSwarmBestPositions(j, :, i) = subSwarmPositions(j, :, i);
        subSwarmBestFitness(j, i) = fitness;

        % 更新全局最优解
        if fitness < globalBestFitness
            globalBestFitness = fitness;
            globalBestPosition = subSwarmPositions(j, :, i);
        end
    end
end

%% 冷启动参数
coldStartFrequency = 10; % 每隔若干迭代进行冷启动
resetRatio = 0.7; % 重置粒子比例

%% 迭代过程
for iter = 1:maxIterations
    for i = 1:numSubSwarms
        for j = 1:popSize
            % 更新速度
             wa = w_max - (w_max - w_min) * (iter / maxIterations);

    % 动态调整认知和社会加速度系数
    c1 = c1_initial + ((2.5 - c1_initial) * (iter / maxIterations)); % c1从1.5增加到2.5
    c2 = c2_initial - ((c2_initial - 1.0) * (iter / maxIterations)); % c2从1.5减小到1.0
            r1 = rand(1, D);
            r2 = rand(1, D);
            subSwarmVelocities(j, :, i) = wa * subSwarmVelocities(j, :, i) + ...
                                          c1 * r1 .* (subSwarmBestPositions(j, :, i) - subSwarmPositions(j, :, i)) + ...
                                          c2 * r2 .* (globalBestPosition - subSwarmPositions(j, :, i));

            % 更新位置
            subSwarmPositions(j, :, i) = subSwarmPositions(j, :, i) + subSwarmVelocities(j, :, i);

            % 保证参数在定义域范围内
            subSwarmPositions(j, :, i) = max(subSwarmPositions(j, :, i), DefineDomain(1,:));
            subSwarmPositions(j, :, i) = min(subSwarmPositions(j, :, i), DefineDomain(2,:));

            % 计算新的适应度
            fitness = sqrt(mean((convolution1(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5)) - fit).^2));

            % 更新个体最优解
            if fitness < subSwarmBestFitness(j, i)
                subSwarmBestFitness(j, i) = fitness;
                subSwarmBestPositions(j, :, i) = subSwarmPositions(j, :, i);
            end

            % 更新全局最优解
            if fitness < globalBestFitness
                globalBestFitness = fitness;
                globalBestPosition = subSwarmPositions(j, :, i);
            end
        end
    end

    % 冷启动
    if mod(iter, coldStartFrequency) == 0
        fprintf('Cold start triggered at iteration %d\n', iter);
        for i = 1:numSubSwarms
            numReset = floor(popSize * resetRatio);
            for resetIdx = 1:numReset
                idx = randi([1, popSize]);
                subSwarmPositions(idx, :, i) = DefineDomain(1,:) + rand(1, D) .* (DefineDomain(2,:) - DefineDomain(1,:));
                subSwarmVelocities(idx, :, i) = zeros(1, D);
            end
        end
    end

    fitnessHistory(iter) = globalBestFitness;
    fprintf('Iteration %d: Global Best Fitness = %.4f\n', iter, globalBestFitness);
end

%% 绘制收敛曲线
figure;
plot(1:maxIterations, fitnessHistory, 'o-');
xlabel('Iteration');
ylabel('Global Best Fitness');
title('Convergence Curve');
grid on;

%% 输出全局最优参数
w = globalBestPosition(1);
lambda = globalBestPosition(2);
sigema_0 = globalBestPosition(3);
sigema_k = globalBestPosition(4);
sigema_c = globalBestPosition(5);
error = globalBestFitness;
% 输出原始和优化后的参数值
output_str = sprintf(['Original and Optimized Parameters Comparison:\n\n' ...
                      'Original w = %.4f, Optimized w = %.4f\n' ...
                      'Original lambda = %.4f, Optimized lambda = %.4f\n' ...
                      'Original sigema_0 = %.4f, Optimized sigema_0 = %.4f\n' ...
                      'Original sigema_k = %.4f, Optimized sigema_k = %.4f\n' ...
                      'Original sigema_c = %.4f, Optimized sigema_c = %.4f\n' ...
                      'Error = %.4f\n'], ...
                      original_w, w, ...
                      original_lambda, lambda, ...
                      original_sigema_0, sigema_0, ...
                      original_sigema_k, sigema_k, ...
                      original_sigema_c, sigema_c, ...
                      error);
fprintf('%s', output_str);



%% 结果对比
result = convolution1(fit, w, lambda, sigema_0, sigema_k, sigema_c);
error = sqrt(mean((result - fit).^2));

figure;
x = 1:length(fit); % 横轴点数
plot(x, fit, '-b', 'DisplayName', 'Smoothed Input'); % 绘制平滑后的Input图像
hold on;
plot(x, result, '-r', 'DisplayName', 'Result'); % 绘制Result图像
plot(x, fit - result, '-g', 'DisplayName', 'Difference'); % 绘制差值图像
xlabel('Points'); % 横轴标签
ylabel('Values'); % 纵轴标签
legend; % 显示图例
title('Comparison of Smoothed Input and Result with Difference'); % 图像标题
grid on;
hold off

toc;

