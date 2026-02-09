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
fixedCPSO_resetRatio = 0.3;         % 粒子重置比例

%% ============== 步骤1：原有核心代码（数据处理+卷积逻辑，无修改） ==============
inddis=0.5;

% 原始MRI参数
w = 1;
lambda =1;
sigema_0=0.8;
sigema_k=0;
sigema_c=0;
original_w = w;
original_lambda = lambda;
original_sigema_0 = sigema_0;
original_sigema_k = sigema_k;
original_sigema_c = sigema_c;

obj_tn = 10*ones(1,6);%每层"xx"nm*（1，xx层）
obj_tn=[obj_tn 15];
cons=5;

% sigma变化函数（常数，无修改）
sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0;

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
result2(result2>1)=1;
result2(result2<0)=0;
fit=result2;
DataLength = length(fit);  % 记录输入的实验数据数组的大小

% 定义MRI参数定义域（仅用于CPSO优化约束，无修改）
DefineDomain = [0 0 0 0 0;
                3 3 3 0 0]; % w, lambda, sigema_0, sigema_k, sigema_c
D = 5; % MRI参数维度
maxIterations = 200; % CPSO最大迭代次数（固定）
convergeThreshold = 0.01; % 收敛阈值（固定）

%% ============== 步骤2：Morris筛选法配置（核心修改2：分析MRI 3个参数） ==============
% 2.1 定义待分析的MRI参数信息（3个参数，名称+上下界）
numSensParams = 3; % 从7个改为3个（MRI核心参数）
sensParamNames = {'w', 'lambda', 'sigema_0'}; % 灵敏度分析对象切换
% MRI参数下界（参考原始值，合理缩小范围，保证分析有效性）
sensParamLower = [0.5, 0.5, 0.5];
% MRI参数上界（参考DefineDomain，合理扩大范围，保证分析有效性）
sensParamUpper = [3, 3, 3];
% Morris实验设计参数（不变，平衡计算量和稳定性）
numTrajectories = 4; % 轨迹数
numLevels = 6; % 每个参数的等间距水平数
delta = numLevels / (2 * (numLevels - 1)); % Morris设计核心参数

% 2.2 生成Morris样本集（3维，对应MRI 3个参数）
fprintf('\nGenerating Morris sample set for MRI parameters...\n');
morrisDesign = [];
for traj = 1:numTrajectories
    % 生成初始点（3维，对应3个MRI参数）
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

%% ============== 步骤3：批量运行CPSO（核心修改3：固定CPSO参数，遍历MRI样本） ==============
fprintf('\nStarting batch CPSO run for MRI parameters (%d samples total)...\n', numSamples);
for sampleIdx = 1:numSamples
    % 提取当前Morris样本的MRI 3个参数（用于初始化CPSO的优化起点）
    currentMRI_w = morrisSamples(sampleIdx, 1);
    currentMRI_lambda = morrisSamples(sampleIdx, 2);
    currentMRI_sigema0 = morrisSamples(sampleIdx, 3);
    
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
    
    % 粒子初始化（使用当前MRI样本参数作为初始值参考）
    for i = 1:fixedCPSO_numSubSwarms
        for j = 1:fixedCPSO_popSize
            % 初始化MRI核心参数（从Morris样本取值）
            subSwarmPositions(j, 1, i) = currentMRI_w + randn(1)*0.1; % w
            subSwarmPositions(j, 2, i) = currentMRI_lambda + randn(1)*0.1; % lambda
            subSwarmPositions(j, 3, i) = currentMRI_sigema0 + randn(1)*0.1; % sigema_0
            % 固定其他参数（sigema_k, sigema_c）
            subSwarmPositions(j, 4, i) = sigema_k;
            subSwarmPositions(j, 5, i) = sigema_c;
            
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

%% ============== 步骤4：计算Morris灵敏度指数（核心修改4：对应MRI 3个参数） ==============
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

% 整理灵敏度结果表格（3个MRI参数）
sensitivityResult = table(...
    sensParamNames', mu', sigma', mu_star', ...
    'VariableNames', {'MRI_ParamName', 'MainEffect_mu', 'InteractionEffect_sigma', 'TotalEffect_mu_star'});
% 按总效应排序（从高到低）
sensitivityResult = sortrows(sensitivityResult, 'TotalEffect_mu_star', 'descend');

% 打印灵敏度排序结果
fprintf('\n==================== MRI Parameters Sensitivity Ranking ====================\n');
disp(sensitivityResult);

%% ============== 步骤5：灵敏度分析可视化（核心修改5：适配3个MRI参数） ==============
% 5.1 图1：主效应vs交互效应散点图
figure('Position', [100, 100, 800, 600]);
scatter(mu, sigma, 200, mu_star, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Main Effect (\mu)');
ylabel('Interaction Effect (\sigma)');
title('Morris Global Sensitivity Analysis: MRI Core Parameters');
cbar = colorbar;
ylabel(cbar, 'Total Effect (\mu^*)');
grid on;
% 添加MRI参数标签
for i = 1:numSensParams
    text(mu(i)+0.001, sigma(i)+0.001, sensParamNames{i}, 'FontSize', 12, 'FontWeight', 'bold');
end

% 5.2 图2：灵敏度总效应柱状图（3个参数，排序展示）
figure('Position', [200, 200, 700, 400]);
paramLabels = sensitivityResult.MRI_ParamName;
totalEffect = sensitivityResult.TotalEffect_mu_star;
barh(1:numSensParams, totalEffect, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
set(gca, 'YTickLabel', paramLabels);
xlabel('Total Sensitivity Effect (\mu^*)');
ylabel('MRI Core Parameters');
title('MRI Parameters Sensitivity Ranking (Total Effect)');
grid on;
% 标注数值
for i = 1:numSensParams
    text(totalEffect(i)+0.0005, i, sprintf('%.4f', totalEffect(i)), 'FontSize', 10, 'VerticalAlignment', 'middle');
end

% 5.3 图3：MRI参数的RMSE影响趋势图（3个子图，1行3列）
figure('Position', [300, 300, 1200, 400]);
layoutRow = 1; % 1行
layoutCol = 3; % 3列（适配3个MRI参数）

for idx = 1:numSensParams
    % 提取当前MRI参数的信息
    paramRank = idx;
    paramName = sensitivityResult.MRI_ParamName{paramRank};
    paramCol = find(strcmp(sensParamNames, paramName));
    paramValues = morrisSamples(:, paramCol);
    paramRMSE = sampleResults(:, 1);
    
    % 子图布局
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
sgtitle('RMSE Influence Trend of MRI Core Parameters (Sorted by Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');

%% ============== 步骤6：原有结果输出和对比（无修改） ==============
% % 提取最优MRI参数
% w = globalBestPosition(1);
% lambda = globalBestPosition(2);
% sigema_0 = globalBestPosition(3);
% sigema_k = globalBestPosition(4);
% sigema_c = globalBestPosition(5);
% error = globalBestFitness;
% 
% % 输出参数对比
% output_str = sprintf(['\nOriginal and Optimized MRI Parameters Comparison:\n\n' ...
%                       'Original w = %.4f, Optimized w = %.4f\n' ...
%                       'Original lambda = %.4f, Optimized lambda = %.4f\n' ...
%                       'Original sigema_0 = %.4f, Optimized sigema_0 = %.4f\n' ...
%                       'Original sigema_k = %.4f, Optimized sigema_k = %.4f\n' ...
%                       'Original sigema_c = %.4f, Optimized sigema_c = %.4f\n' ...
%                       'Final Error = %.4f\n'], ...
%                       original_w, w, ...
%                       original_lambda, lambda, ...
%                       original_sigema_0, sigema_0, ...
%                       original_sigema_k, sigema_k, ...
%                       original_sigema_c, sigema_c, ...
%                       error);
% fprintf('%s', output_str);
% 
% % 结果对比图
% result = convolution(fit, w, lambda, sigema_0, sigema_k, sigema_c);
% figure;
% x = 1:length(fit);
% plot(x, fit, '-b', 'DisplayName', 'Smoothed Input');
% hold on;
% plot(x, result, '-r', 'DisplayName', 'CPSO Fitting Result (Fixed CPSO Params)');
% plot(x, fit - result, '-g', 'DisplayName', 'Difference (Input - Result)');
% xlabel('Points');
% ylabel('Values');
% legend('Location', 'best');
% title('Comparison of Smoothed Input and CPSO Fitting Result (Fixed CPSO Params)');
% grid on;
% hold off;
% ============== 步骤1：初始化容器，存放所有参数的横/纵坐标数据 ==============
% 定义一个结构体数组，每个元素对应一个参数的完整数据（方便管理）
paramDataStruct = struct('ParamName', '', 'Rank', 0, 'X_Data', [], 'Y_Data', []);
paramDataStruct = repmat(paramDataStruct, numSensParams, 1); % 复制为7个参数的结构体
% 5.3 图3：所有参数的RMSE影响趋势图（每个参数独立子图，同时提取数据）
figure('Position', [300, 300, 1400, 600]);
numAllParams = numSensParams;
layoutRow = 2;
layoutCol = 4;

for idx = 1:numAllParams
    % 提取当前参数的信息
    paramRank = idx;
    paramName = sensitivityResult.ParamName{paramRank};
    paramCol = find(strcmp(sensParamNames, paramName));
    paramValues = morrisSamples(:, paramCol); % 横坐标数据：当前参数的所有取值
    paramRMSE = sampleResults(:, 1); % 纵坐标数据：对应的RMSE值（所有样本的RMSE）
    
    % ========== 核心：将当前参数的数据存入结构体 ==========
    paramDataStruct(idx).ParamName = paramName; % 参数名
    paramDataStruct(idx).Rank = paramRank; % 灵敏度排名
    paramDataStruct(idx).X_Data = paramValues; % 横坐标数据（列向量，方便后续处理）
    paramDataStruct(idx).Y_Data = paramRMSE; % 纵坐标数据（列向量，和横坐标一一对应）
    
    % 绘制趋势图（原有逻辑不变，仅增加上述数据提取）
    subplot(layoutRow, layoutCol, idx);
    scatter(paramValues, paramRMSE, 50, 'b', 'filled');
    xlabel(paramName);
    ylabel('Final RMSE');
    title(['Rank ', num2str(idx), ': ', paramName]);
    grid on;
    p = polyfit(paramValues, paramRMSE, 1);
    hold on;
    plot(paramValues, polyval(p, paramValues), 'r-', 'LineWidth', 1.5);
    hold off;
end

% 调整子图间距
sgtitle('RMSE Influence Trend of All CPSO Parameters (Sorted by Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');
% 总耗时
fprintf('\nTotal runtime: %.2fs\n', toc());
fprintf('\nCreating separate variables for Figure 3 in MATLAB workspace...\n');

% 遍历3个MRI参数，在工作区创建独立变量
for idx = 1:numSensParams
    % 1. 提取当前参数信息
    paramName = sensitivityResult.MRI_ParamName{idx}; % 当前参数名称（w/lambda/sigema_0）
    paramCol = find(strcmp(sensParamNames, paramName)); % 对应样本列索引
    
    % 2. 提取核心数据（横坐标+纵坐标，对应图3子图）
    param_x_data = morrisSamples(:, paramCol); % 横坐标：参数取值
    param_y_data = sampleResults(:, 1); % 纵坐标：最终RMSE（核心需求）
    
    % 3. 为当前参数创建独立的工作区变量（关键：assignin函数，直接写入工作区）
    % 横坐标变量：如 w_x, lambda_x, sigema_0_x
    x_var_name = sprintf('%s_x', paramName);
    assignin('base', x_var_name, param_x_data);
    
    % 纵坐标变量：如 w_y, lambda_y, sigema_0_y（你重点需要的RMSE数据）
    y_var_name = sprintf('%s_y', paramName);
    assignin('base', y_var_name, param_y_data);
    
    % （可选）创建趋势线数据变量：如 w_trend, lambda_trend, sigema_0_trend
    p_fit = polyfit(param_x_data, param_y_data, 1);
    param_trend_data = polyval(p_fit, param_x_data);
    trend_var_name = sprintf('%s_trend', paramName);
    assignin('base', trend_var_name, param_trend_data);
    
    % 打印提示：告知工作区中生成的变量名
    fprintf('  Workspace variables created for %s: %s (X), %s (Y), %s (Trend)\n', ...
        paramName, x_var_name, y_var_name, trend_var_name);
end

fprintf('All Figure 3 data variables are ready in MATLAB workspace!\n');
