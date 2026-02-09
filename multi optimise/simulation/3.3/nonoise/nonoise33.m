clear
clc
tic
%% 导入数据
noiselevel=0;%噪声范围
inddis=0.5;
w = 1;
lambda =1;
sigema_0=2;
sigema_k=0.5;
sigema_c=0.05;
% sigema=sigema_k*sigema_0^(sigema_c*(i*inddis));
original_w = w;
original_lambda = lambda;
original_sigema_0 = sigema_0;
original_sigema_k = sigema_k;
original_sigema_c = sigema_c;
%构造膜层结构
% obj_tn=[16 14.5 15 15 15 15 15 15];
% obj_tn=[obj_tn 10.388];
%  obj_tn = 3*ones(1,6);%每层"xx"nm*（1，xx层）
%  obj_tn=[obj_tn 5];
obj_tn = 10*ones(1,6);%每层"xx"nm*（1，xx层）
% obj_tn=[16 14.5 15 15 15 15 15 15];
obj_tn=[obj_tn 15];
cons=5;


%------------------------------------------------------------------------------------------------------------------------
%% sigma
%线性
%  sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0+sigema_k*(i*inddis);
%常数
%  sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0;
%开根号
 %sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sqrt(sigema_0+sigema_k*(i*inddis)^2);
%指数
 sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_k*sigema_0^(sigema_c*(i*inddis));
%------------------------------------------------------------------------------------------------------------------------
%% 卷积
wide=sum(obj_tn);
% tn = round(obj_tn);  %四舍五入
tn=obj_tn;
layer=length(tn);
z0=zeros(1,layer+1);
for i=1:length(z0)       %分层位置
    if i==1
        z0(1,i)=0; %起始层
    else
        z0(1,i)=z0(1,1)+sum(tn(1:i-1));
    end
end

%x1为深度，y为成分分布
x1=z0(1):inddis:z0(end);
y=zeros(1,round((z0(end)-z0(1))/inddis+1));
if length(x1) == length(y)
    y=y;
else
    y=zeros(1,round((z0(end)-z0(1))/inddis));
end


for i=1:1:layer               %建立层结构——奇数浓度为0，偶数浓度为1，层结构奇偶交错
    len=round((z0(i+1)-z0(i))/inddis);
    if mod(i,2)==1     %奇数层膜层时
        for k=1:1:len
            y(round((z0(i)-z0(1))/inddis)+k)=0;%第一层是0浓度层这边就是0，否则是1
        end
    else %偶数
        for k=1:1:len
            y(round((z0(i)-z0(1))/inddis)+k)=1;
        end
    end
end


ra2=x1';
y=y';
depth_data=y;
n=length(ra2);
c=zeros(n);


%卷积
%卷积
for i=1:1:length(ra2)
    sigema=sigma_change(sigema_0,sigema_k,i,inddis,sigema_c);
     delta=4;
    sigwx=-w:inddis:cons*delta;
    gw=exp(-(sigwx+w)/w)/w;
    sigsx=-cons*delta:inddis:cons*delta;
    gsigema=exp(-sigsx.^2/(2*(sigema.^2)))/(sqrt(2*pi)*sigema);
    siglx=-cons*delta:inddis:0;
    glambda=exp(siglx/lambda)/lambda;
    sigy=conv(conv(gw,gsigema),glambda);
    if  floor((cons*delta+w)/inddis)==0
        sigy=sigy(ceil((cons*delta+w)/inddis):floor((3*cons*delta+w)/inddis));
    else
        sigy=sigy(floor((1*cons*delta+w)/inddis):floor((3*cons*delta+w)/inddis));%取样
    end
    tms=sum(sigy)*inddis;
    sigy=sigy./tms;
    sigy=fliplr(sigy);%∫gXdz的时候分辨率函数和X还要进行一次卷积（翻转）
    
    if i<=cons*delta/inddis
        for k=1:1:floor((length(sigy)+1)/2-1+i)
            c(i,k)=sigy(floor((length(sigy)+1)/2)-i+k);
        end
    elseif i>floor(cons*delta/inddis) && i<floor(n-cons*delta/inddis)
        for k=1:1:2*cons*delta/inddis
            c(i,i-floor(cons*delta/inddis)+k)=sigy(k);
        end
    else
        for k=1:1:n-i+floor(cons*delta/inddis)
            c(i,i-floor(cons*delta/inddis)+k)=sigy(k);
        end
    end
end
result1=c*depth_data*inddis;%c*depth=∫gX,缺dz，所以result=c*depth_data*inddis
noise=noiselevel*randn(size(result1));%高斯噪声
result2 = result1 + noise;
fit1=result1;
result2(result2>1)=1;
result2(result2<0)=0;

fit=result2;
DataLength = length(fit);  % 记录输入的实验数据数组的大小
%% 新增：智能计算目标层（2、4层）的有效索引范围（仅用于最后结果对比，不影响优化）
fprintf('开始智能推导目标层索引范围...\n');

% 1. 配置目标层（可灵活修改）
target_layers = [2,3,4,5,6]; % 你的需求：第2、4层
obj_layer_num = length(obj_tn); % 膜层总层数

% 2. 异常检查
if any(target_layers > obj_layer_num) || any(target_layers < 1)
    error('目标层编号超出膜层总层数！膜层总层数为%d，请修正target_layers。', obj_layer_num);
end

% 3. 计算各层累计物理深度
cum_depth = [0, cumsum(obj_tn)]; % [0, 层1厚度, 层1+层2厚度, ..., 总厚度]
valid_indices = []; % 存储目标层对应的data/ra2_ex索引（对应fit的前length(data)个元素）

% 4. 遍历目标层，推导有效索引
for idx = 1:length(target_layers)
    current_layer = target_layers(idx);
    % 该层物理深度区间
    layer_depth_start = cum_depth(current_layer);
    layer_depth_end = cum_depth(current_layer+1);
    fprintf('第%d层：物理深度区间[%f, %f]\n', current_layer, layer_depth_start, layer_depth_end);
    
    % 映射为ra2_ex/data的索引（避免下标越界）
    k_start = find(ra2 >= layer_depth_start, 1, 'first');
    k_end = find(ra2 <= layer_depth_end, 1, 'last');
    if isempty(k_start); k_start = length(ra2); end
    if isempty(k_end); k_end = 1; end
    k_start = max(1, k_start);
    k_end = min(length(ra2), k_end);
    
    % 收集有效索引
    layer_indices = k_start:k_end;
    valid_indices = [valid_indices, layer_indices];
end

% 去重排序，得到最终有效索引（对应data/ra2_ex，也对应fit(1:end-1)）
valid_indices = unique(sort(valid_indices));
fprintf('目标层有效索引范围：[%d, %d]，共%d个数据点\n', ...
    min(valid_indices), max(valid_indices), length(valid_indices));
fprintf('----------------------------------------\n');


%% 定义参数范围
DefineDomain = [0 0 0 0 0;
                3 3 3 1 1]; % w, lambda, sigema_0, sigema_k, sigema_c，参数范围上下限可自定义，不影响结果，影响时间

%% CPSO参数设置
numSubSwarms = 10; % 子群数量  20、 25、50、100    时间增加，线性效果较好，一般50以上的选取迭代次数即可选择在100内
popSize = 5; % 每个子群的粒子数量 20、25、50、100 
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
        fitness = sqrt(mean((convolution(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5)) - fit).^2));
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
coldStartFrequency =10; % 每隔若干迭代进行冷启动 
resetRatio = 0.7; % 重置粒子比例   调整为0.5-0.9即可

%% 迭代过程
for iter = 1:maxIterations
    for i = 1:numSubSwarms
        for j = 1:popSize
            % 更新速度
            r1 = rand(1, D);
            r2 = rand(1, D);
                         wa = w_max - (w_max - w_min) * (iter / maxIterations);
% 
%     % 动态调整认知和社会加速度系数
    c1 = c1_initial + ((2.5 - c1_initial) * (iter / maxIterations)); % c1从1.5增加到2.5
    c2 = c2_initial - ((c2_initial - 1.0) * (iter / maxIterations)); % c2从1.5减小到1.0
            subSwarmVelocities(j, :, i) = wa * subSwarmVelocities(j, :, i) + ...
                                          c1 * r1 .* (subSwarmBestPositions(j, :, i) - subSwarmPositions(j, :, i)) + ...
                                          c2 * r2 .* (globalBestPosition - subSwarmPositions(j, :, i));

            % 更新位置
            subSwarmPositions(j, :, i) = subSwarmPositions(j, :, i) + subSwarmVelocities(j, :, i);

            % 保证参数在定义域范围内
            subSwarmPositions(j, :, i) = max(subSwarmPositions(j, :, i), DefineDomain(1,:));
            subSwarmPositions(j, :, i) = min(subSwarmPositions(j, :, i), DefineDomain(2,:));

            % 计算新的适应度
            fitness = sqrt(mean((convolution(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5)) - fit).^2));

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
% Output original and optimized parameters with error in scientific notation
output_str = sprintf(['Original and Optimized Parameters:\n\n' ...
    'Original w = %.4f, Optimized w = %.4f\n' ...
    'Original lambda = %.4f, Optimized lambda = %.4f\n' ...
    'Original sigema_0 = %.4f, Optimized sigema_0 = %.4f\n' ...
    'Original sigema_k = %.4f, Optimized sigema_k = %.4f\n' ...
    'Original sigema_c = %.4f, Optimized sigema_c = %.4f\n' ...
    'Error = %.4e\n'], ... % Use %.4e for scientific notation
    original_w, w, original_lambda, lambda, ...
    original_sigema_0, sigema_0, original_sigema_k, ...
    sigema_k, original_sigema_c, sigema_c, error);
fprintf('%s', output_str);


%% 结果对比
%% 结果对比
result = convolution(fit, w, lambda, sigema_0, sigema_k, sigema_c);
error = sqrt(mean((result - fit1).^2));

figure;
x = 1:length(fit); % 横轴点数
plot(x, fit, '-b', 'DisplayName', 'Smoothed Input'); % 绘制平滑后的Input图像
hold on;
plot(x, result, '-r', 'DisplayName', 'Result'); % 绘制Result图像
plot(x, fit1 - result, '-g', 'DisplayName', 'Difference'); % 绘制差值图像
xlabel('Points'); % 横轴标签
ylabel('Values'); % 纵轴标签
legend; % 显示图例
title('Comparison of Smoothed Input and Result with Difference'); % 图像标题
grid on;
hold off;
% 2. 新增：目标层数据提取与误差计算（核心需求）
% 提取fit和result中对应目标层的有效数据（排除fit最后补的0，对应valid_indices）
fit_target = fit1(valid_indices);
result_target = result(valid_indices);
difftarget=fit_target-result_target;
% 计算目标层的RMSE误差（和全部数据误差计算方式一致）
error_target = sqrt(mean((fit_target - result_target).^2));

% 原对比图绘制 + 目标层数据高亮（核心改动）
figure;
x = 1:length(fit); % 横轴点数
% 1. 绘制全部数据的三条曲线（原逻辑）
plot(x, fit, '-b', 'DisplayName', 'Smoothed Input');
hold on;
plot(x, result, '-r', 'DisplayName', 'Result');
plot(x, fit1 - result, '-g', 'DisplayName', 'Difference');

% 2. 新增：绘制目标层数据的高亮标记（不同颜色+大尺寸标记，突出显示）
plot(valid_indices, fit1(valid_indices), 'bs', 'MarkerSize', 6, 'DisplayName', 'Target Layer Input');
plot(valid_indices, result(valid_indices), 'rd', 'MarkerSize', 6, 'DisplayName', 'Target Layer Result');
% 修正报错：先计算差值数组，再索引（推荐写法，更易读）
diff_all = fit1 - result;
plot(valid_indices, diff_all(valid_indices), 'gv', 'MarkerSize', 6, 'DisplayName', 'Target Layer Difference');

% 3. 原图标注（保留）
xlabel('Points'); % 横轴标签
ylabel('Values'); % 纵轴标签
legend; % 显示图例（包含高亮数据）
title('Comparison of Smoothed Input and Result with Target Layer Highlight'); % 标题微调，标注高亮
grid on;
hold off;

% 4. 新增：输出目标层误差结果
fprintf('----------------------------------------\n');
fprintf('Target Layers (2,4) Error = %.4f\n', error_target);
fprintf('Target Layers Data Count = %d\n', length(valid_indices));
toc;
