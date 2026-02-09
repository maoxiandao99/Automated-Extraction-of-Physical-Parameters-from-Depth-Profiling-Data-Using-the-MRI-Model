clear
clc
tic

%% ============== 步骤0：补充遗漏参数（避免报错） + 固定CPSO参数（核心修改1） ==============
% 补充噪声水平定义（原有代码遗漏，避免运行报错）
noiselevel = 0; % 高斯噪声水平，可根据需求调整

% 固定CPSO参数（经典稳定值，不再参与灵敏度分析，全程不变）
fixedCPSO_numSubSwarms = 10;        % 子群数量
fixedCPSO_popSize = 5;              % 每个子群粒子数
fixedCPSO_wa = 0.5;                 % 惯性权重
fixedCPSO_c1 = 1.5;                 % 自我认知学习因子
fixedCPSO_c2 = 1.5;                 % 社会认知学习因子
fixedCPSO_coldStartFrequency = 5;   % 冷启动频率
fixedCPSO_resetRatio = 0.7;         % 粒子重置比例

%% ============== 步骤1：原有核心代码（数据处理+卷积逻辑，无修改） ==============
inddis=0.5;

% 原始MRI参数
w = 1;
lambda =1;
sigema_0=2;
sigema_k=0.5;
sigema_c=0.05;
original_w = w;
original_lambda = lambda;
original_sigema_0 = sigema_0;
original_sigema_k = sigema_k;
original_sigema_c = sigema_c;

obj_tn = 10*ones(1,6);%每层"xx"nm*（1，xx层）
obj_tn=[obj_tn 15];
cons=5;
sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_k*sigema_0^(sigema_c*(i*inddis)); % 保留你的新指数公式

%% 卷积逻辑（无修改）
wide=sum(obj_tn);
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
            y(round((z0(i)-z0(1))/inddis)+k)=0;
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

%卷积计算
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
        sigy=sigy(ceil((cons*delta+w)/inddis):floor((3*cons*delta+w)/inddis));%取样
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
result2(result2>1)=1;
result2(result2<0)=0;
fit=result2;
DataLength = length(fit);  % 记录输入的实验数据数组的大小

% 定义MRI参数定义域（仅用于CPSO优化约束，无修改）
DefineDomain = [0 0 0 0 0;
                3 3 3 1 1]; % w, lambda, sigema_0, sigema_k, sigema_c
D = 5; % MRI参数维度
maxIterations = 200; % CPSO最大迭代次数（固定）
convergeThreshold = 0.01; % 收敛阈值（固定）

%% ============== 步骤2：Morris筛选法配置（核心修改2：分析MRI 5个参数，新增sigema_c） ==============
% 2.1 定义待分析的MRI参数信息（5个参数，名称+上下界，新增sigema_c）
numSensParams = 5; % 从4个改为5个（新增sigema_c）
sensParamNames = {'w', 'lambda', 'sigema_0', 'sigema_k', 'sigema_c'}; % 加入sigema_c，统一命名
% MRI参数下界（参考原始值+DefineDomain，补充sigema_c下界）
sensParamLower = [0.5, 0.5, 0.5, 0.0, 0.0]; % 第5项：sigema_c下界=0.0
% MRI参数上界（参考原始值+DefineDomain，补充sigema_c上界）
sensParamUpper = [3, 3, 3, 1, 1.0]; % 第5项：sigema_c上界=1.0（匹配DefineDomain）
% Morris实验设计参数（不变，平衡计算量和稳定性）
numTrajectories = 4; % 轨迹数
numLevels = 6; % 每个参数的等间距水平数
delta = numLevels / (2 * (numLevels - 1)); % Morris设计核心参数

% 2.2 生成Morris样本集（5维，对应MRI 5个参数）
fprintf('\nGenerating Morris sample set for MRI parameters...\n');
morrisDesign = [];
for traj = 1:numTrajectories
    % 生成初始点（5维，对应5个MRI参数，新增sigema_c维度）
    x0 = randi([0, numLevels-2], 1, numSensParams);
    % 生成轨迹
    for p = 1:numSensParams
        x = x0;
        x(p) = x(p) + 1;
        morrisDesign = [morrisDesign; x0; x];
    end
end
morrisDesign = morrisDesign / (numLevels - 1); % 归一化处理
numSamples = size(morrisDesign, 1);

% 2.3 将Morris样本映射到MRI参数的实际取值范围
morrisSamples = zeros(numSamples, numSensParams);
for i = 1:numSensParams
    % 归一化到[0,1]
    morrisSamples(:, i) = (morrisDesign(:, i) - min(morrisDesign(:, i))) / (max(morrisDesign(:, i)) - min(morrisDesign(:, i)));
    % 映射到MRI参数实际范围
    morrisSamples(:, i) = sensParamLower(i) + morrisSamples(:, i) .* (sensParamUpper(i) - sensParamLower(i));
end

% 2.4 初始化结果存储矩阵
sampleResults = zeros(numSamples, 3); % 每列：最终RMSE、收敛迭代数、运行耗时

%% ============== 步骤3：批量运行CPSO（核心修改3：固定CPSO参数，遍历MRI 5个参数样本） ==============
fprintf('\nStarting batch CPSO run for MRI parameters (%d samples total)...\n', numSamples);
for sampleIdx = 1:numSamples
    % 提取当前Morris样本的MRI 5个参数（新增sigema_c，从第5列提取）
    currentMRI_w = morrisSamples(sampleIdx, 1);
    currentMRI_lambda = morrisSamples(sampleIdx, 2);
    currentMRI_sigema0 = morrisSamples(sampleIdx, 3);
    currentMRI_sigmak = morrisSamples(sampleIdx, 4);
    currentMRI_sigmac = morrisSamples(sampleIdx, 5); % 新增：提取sigema_c样本值
    
    % 固定随机种子，消除随机性影响
    rng(123 + sampleIdx);
    
    % 启动计时
    tic();
    
    % ========== 初始化CPSO粒子（使用固定CPSO参数，MRI参数作为优化约束） ==========
    subSwarmPositions = zeros(fixedCPSO_popSize, D, fixedCPSO_numSubSwarms);
    subSwarmVelocities = zeros(fixedCPSO_popSize, D, fixedCPSO_numSubSwarms);
    subSwarmBestPositions = zeros(fixedCPSO_popSize, D, fixedCPSO_numSubSwarms);
    subSwarmBestFitness = Inf(fixedCPSO_popSize, fixedCPSO_numSubSwarms);
    globalBestPosition = zeros(1, D);
    globalBestFitness = Inf;
    fitnessHistory = zeros(maxIterations, 1);
    
    % 粒子初始化（使用当前MRI样本参数作为初始值参考，新增sigema_c初始化）
    for i = 1:fixedCPSO_numSubSwarms
        for j = 1:fixedCPSO_popSize
            % 初始化MRI核心参数（从Morris样本取值，新增sigema_c）
            subSwarmPositions(j, 1, i) = currentMRI_w + randn(1)*0.1; % w
            subSwarmPositions(j, 2, i) = currentMRI_lambda + randn(1)*0.1; % lambda
            subSwarmPositions(j, 3, i) = currentMRI_sigema0 + randn(1)*0.1; % sigema_0
            subSwarmPositions(j, 4, i) = currentMRI_sigmak + randn(1)*0.01; % sigema_k（小扰动）
            subSwarmPositions(j, 5, i) = currentMRI_sigmac + randn(1)*0.005; % 新增：sigema_c（更小扰动，匹配[0,1]范围）
            
            % 限制定义域
            subSwarmPositions(j, :, i) = max(subSwarmPositions(j, :, i), DefineDomain(1,:));
            subSwarmPositions(j, :, i) = min(subSwarmPositions(j, :, i), DefineDomain(2,:));
            subSwarmVelocities(j, :, i) = zeros(1, D);
            
            % 计算初始适应度（卷积+RMSE）
            conv_result = convolution(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5));
            fitness = sqrt(mean((conv_result - fit).^2));
            subSwarmBestPositions(j, :, i) = subSwarmPositions(j, :, i);
            subSwarmBestFitness(j, i) = fitness;
            
            % 更新全局最优
            if fitness < globalBestFitness
                globalBestFitness = fitness;
                globalBestPosition = subSwarmPositions(j, :, i);
            end
        end
    end
    
    % ========== CPSO迭代过程（使用固定CPSO参数，无修改） ==========
    for iter = 1:maxIterations
        for i = 1:fixedCPSO_numSubSwarms
            for j = 1:fixedCPSO_popSize
                % 更新速度（固定CPSO参数）
                r1 = rand(1, D);
                r2 = rand(1, D);
                subSwarmVelocities(j, :, i) = fixedCPSO_wa * subSwarmVelocities(j, :, i) + ...
                                              fixedCPSO_c1 * r1 .* (subSwarmBestPositions(j, :, i) - subSwarmPositions(j, :, i)) + ...
                                              fixedCPSO_c2 * r2 .* (globalBestPosition - subSwarmPositions(j, :, i));
                
                % 更新位置并限制定义域
                subSwarmPositions(j, :, i) = subSwarmPositions(j, :, i) + subSwarmVelocities(j, :, i);
                subSwarmPositions(j, :, i) = max(subSwarmPositions(j, :, i), DefineDomain(1,:));
                subSwarmPositions(j, :, i) = min(subSwarmPositions(j, :, i), DefineDomain(2,:));
                
                % 计算新适应度
                conv_result = convolution(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5));
                fitness = sqrt(mean((conv_result - fit).^2));
                
                % 更新个体最优
                if fitness < subSwarmBestFitness(j, i)
                    subSwarmBestFitness(j, i) = fitness;
                    subSwarmBestPositions(j, :, i) = subSwarmPositions(j, :, i);
                end
                
                % 更新全局最优
                if fitness < globalBestFitness
                    globalBestFitness = fitness;
                    globalBestPosition = subSwarmPositions(j, :, i);
                end
            end
        end
        
        % 冷启动（粒子重置，使用固定CPSO参数）
        if mod(iter, fixedCPSO_coldStartFrequency) == 0
            for i = 1:fixedCPSO_numSubSwarms
                numReset = floor(fixedCPSO_popSize * fixedCPSO_resetRatio);
                for resetIdx = 1:numReset
                    idx = randi([1, fixedCPSO_popSize]);
                    subSwarmPositions(idx, :, i) = DefineDomain(1,:) + rand(1, D) .* (DefineDomain(2,:) - DefineDomain(1,:));
                    subSwarmVelocities(idx, :, i) = zeros(1, D);
                end
            end
        end
        
        % 记录收敛历史
        fitnessHistory(iter) = globalBestFitness;
    end
    
    % ========== 计算当前样本的结果指标 ==========
    % 1. 最终RMSE（核心指标）
    finalRMSE = globalBestFitness;
    % 2. 收敛迭代数
    convergeIter = find(fitnessHistory < convergeThreshold, 1, 'first');
    if isempty(convergeIter)
        convergeIter = maxIterations;
    end
    % 3. 运行耗时
    elapsedTime = toc();
    
    % 保存当前样本结果
    sampleResults(sampleIdx, :) = [finalRMSE, convergeIter, elapsedTime];
    
    % 打印运行进度
    fprintf('Sample %d/%d completed: RMSE=%.6f, ConvergeIter=%d, Time=%.2fs\n', ...
        sampleIdx, numSamples, finalRMSE, convergeIter, elapsedTime);
end

%% ============== 步骤4：计算Morris灵敏度指数（核心修改4：对应MRI 5个参数） ==============
fprintf('\nCalculating Morris sensitivity indices for MRI parameters...\n');
% 提取核心输出指标（最终RMSE）
outputMetric = sampleResults(:, 1);

% 计算Morris指数（mu:主效应, sigma:交互效应, mu_star:总效应）
mu = zeros(1, numSensParams);
sigma = zeros(1, numSensParams);
mu_star = zeros(1, numSensParams);

for p = 1:numSensParams
    % 提取该MRI参数的所有取值和对应输出
    paramValues = morrisSamples(:, p);
    [uniqueVals, ~, idx] = unique(paramValues);
    effect = [];
    
    % 计算参数效应（统一行向量，避免维度错误）
    for valIdx = 1:length(uniqueVals)
        val = uniqueVals(valIdx);
        valOutput = outputMetric(idx == valIdx);
        
        if length(valOutput) >= 2
            valDiff = diff(valOutput); % 列向量
            valDiff_row = valDiff(:)'; % 转为行向量
            effect = [effect, valDiff_row];
        end
    end
    
    % 计算灵敏度指数
    if ~isempty(effect)
        mu(p) = mean(effect);
        sigma(p) = std(effect);
        mu_star(p) = mean(abs(effect));
    else
        mu(p) = 0;
        sigma(p) = 0;
        mu_star(p) = 0;
    end
end

% 整理灵敏度结果表格（5个MRI参数）
sensitivityResult = table(...
    sensParamNames', mu', sigma', mu_star', ...
    'VariableNames', {'MRI_ParamName', 'MainEffect_mu', 'InteractionEffect_sigma', 'TotalEffect_mu_star'});
% 按总效应排序（从高到低）
sensitivityResult = sortrows(sensitivityResult, 'TotalEffect_mu_star', 'descend');

% 打印灵敏度排序结果
fprintf('\n==================== MRI Parameters Sensitivity Ranking ====================\n');
disp(sensitivityResult);

%% ============== 步骤5：灵敏度分析可视化（核心修改5：适配MRI 5个参数） ==============
% 5.1 图1：主效应vs交互效应散点图（自动适配5个参数）
figure('Position', [100, 100, 800, 600]);
scatter(mu, sigma, 200, mu_star, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Main Effect (\mu)');
ylabel('Interaction Effect (\sigma)');
title('Morris Global Sensitivity Analysis: MRI Core Parameters (Including sigema_c)');
cbar = colorbar;
ylabel(cbar, 'Total Effect (\mu^*)');
grid on;
% 添加MRI参数标签
for i = 1:numSensParams
    text(mu(i)+0.001, sigma(i)+0.001, sensParamNames{i}, 'FontSize', 12, 'FontWeight', 'bold');
end

% 5.2 图2：灵敏度总效应柱状图（5个参数，排序展示）
figure('Position', [200, 200, 700, 450]); % 略微加高，适配5个参数
paramLabels = sensitivityResult.MRI_ParamName;
totalEffect = sensitivityResult.TotalEffect_mu_star;
barh(1:numSensParams, totalEffect, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
set(gca, 'YTickLabel', paramLabels);
xlabel('Total Sensitivity Effect (\mu^*)');
ylabel('MRI Core Parameters');
title('MRI Parameters Sensitivity Ranking (Total Effect, Including sigema_c)');
grid on;
% 标注数值
for i = 1:numSensParams
    text(totalEffect(i)+0.0005, i, sprintf('%.4f', totalEffect(i)), 'FontSize', 10, 'VerticalAlignment', 'middle');
end

% 5.3 图3：MRI参数的RMSE影响趋势图（5个子图，2行3列，适配新增sigema_c）
figure('Position', [300, 300, 1600, 800]); % 加高加宽，适配2行3列
layoutRow = 2; % 2行
layoutCol = 3; % 3列（完美容纳5个参数，无拥挤）

for idx = 1:numSensParams
    % 提取当前MRI参数的信息
    paramRank = idx;
    paramName = sensitivityResult.MRI_ParamName{paramRank};
    paramCol = find(strcmp(sensParamNames, paramName));
    paramValues = morrisSamples(:, paramCol);
    paramRMSE = sampleResults(:, 1);
    
    % 子图布局（2行3列，逐个填充）
    subplot(layoutRow, layoutCol, idx);
    scatter(paramValues, paramRMSE, 80, 'b', 'filled');
    xlabel(paramName);
    ylabel('Final RMSE');
    title(['Rank ', num2str(idx), ': ', paramName]);
    grid on;
    % 拟合趋势线
    p = polyfit(paramValues, paramRMSE, 1);
    hold on;
    plot(paramValues, polyval(p, paramValues), 'r-', 'LineWidth', 1.5);
    hold off;
end

% 调整子图间距，添加总标题
sgtitle('RMSE Influence Trend of MRI Core Parameters (Including sigema_c, Sorted by Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');

%% ============== 步骤6：图3数据拆分，保存到MATLAB工作区（适配5个参数，新增sigema_c变量） ==============
fprintf('\nCreating separate variables for Figure 3 in MATLAB workspace...\n');

% 遍历5个MRI参数，在工作区创建独立变量（新增sigema_c）
for idx = 1:numSensParams
    % 1. 提取当前参数信息
    paramName = sensitivityResult.MRI_ParamName{idx}; % 当前参数名称（含sigema_c）
    paramCol = find(strcmp(sensParamNames, paramName)); % 对应样本列索引
    
    % 2. 提取核心数据（横坐标+纵坐标，对应图3子图）
    param_x_data = morrisSamples(:, paramCol); % 横坐标：参数取值
    param_y_data = sampleResults(:, 1); % 纵坐标：最终RMSE（核心需求）
    
    % 3. 为当前参数创建独立的工作区变量（关键：assignin函数，直接写入工作区）
    % 横坐标变量：如 w_x, ..., sigema_c_x
    x_var_name = sprintf('%s_x', paramName);
    assignin('base', x_var_name, param_x_data);
    
    % 纵坐标变量：如 w_y, ..., sigema_c_y（你重点需要的RMSE数据）
    y_var_name = sprintf('%s_y', paramName);
    assignin('base', y_var_name, param_y_data);
    
    % （可选）创建趋势线数据变量：如 w_trend, ..., sigema_c_trend
    p_fit = polyfit(param_x_data, param_y_data, 1);
    param_trend_data = polyval(p_fit, param_x_data);
    trend_var_name = sprintf('%s_trend', paramName);
    assignin('base', trend_var_name, param_trend_data);
    
    % 打印提示：告知工作区中生成的变量名
    fprintf('  Workspace variables created for %s: %s (X), %s (Y), %s (Trend)\n', ...
        paramName, x_var_name, y_var_name, trend_var_name);
end

fprintf('All Figure 3 data variables (including sigema_c) are ready in MATLAB workspace!\n');

%% ============== 步骤7：总耗时输出 ==============
fprintf('\nTotal runtime: %.2fs\n', toc());

%% ============== 辅助函数：卷积函数（同步更新sigma_change，确保计算一致） ==============
function conv_result = convolution(fit, w, lambda, sigema_0, sigema_k, sigema_c)
    inddis=0.5;
    cons=5;
    % 同步主代码的sigma_change指数公式，避免计算偏差
    sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_k*sigema_0^(sigema_c*(i*inddis));
    
    % 重建卷积逻辑（与主代码一致）
    obj_tn = 10*ones(1,6);
    obj_tn=[obj_tn 15];
    wide=sum(obj_tn);
    tn=obj_tn;
    layer=length(tn);
    z0=zeros(1,layer+1);
    for i=1:length(z0)
        if i==1
            z0(1,i)=0;
        else
            z0(1,i)=z0(1,1)+sum(tn(1:i-1));
        end
    end
    
    x1=z0(1):inddis:z0(end);
    y=zeros(1,round((z0(end)-z0(1))/inddis+1));
    if length(x1) == length(y)
        y=y;
    else
        y=zeros(1,round((z0(end)-z0(1))/inddis));
    end
    
    for i=1:1:layer
        len=round((z0(i+1)-z0(i))/inddis);
        if mod(i,2)==1
            for k=1:1:len
                y(round((z0(i)-z0(1))/inddis)+k)=0;
            end
        else
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
            sigy=sigy(floor((1*cons*delta+w)/inddis):floor((3*cons*delta+w)/inddis));
        end
        tms=sum(sigy)*inddis;
        sigy=sigy./tms;
        sigy=fliplr(sigy);
        
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
    
    conv_result = c*depth_data*inddis;
    % 匹配fit的长度（避免维度错误）
    if length(conv_result) ~= length(fit)
        conv_result = interp1(1:length(conv_result), conv_result, linspace(1, length(conv_result), length(fit)));
    end
end