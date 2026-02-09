clear
clc
tic

%% ============== 步骤1：保留你原有核心代码（数据处理+CPSO核心逻辑） ==============
inddis=3;

% % w = 2.2;
% % lambda = 1.2;
% % sigema_0=0.2;
% % sigema_k=0;
% % sigema_c=0;
original_w =1.1;
original_lambda = 0.001;
original_sigema_0 = 1.6;
original_sigema_k = 0.004;
original_sigema_c = 0;
%构造膜层结构
obj_tn=[70.0,45.6,68.7,44.6,73.4,47.8,76.6];
obj_tn=[obj_tn 117.11457];
% 
cons=15;


% 定义sigma变化函数（线性，保留你的选择）
sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0+sigema_k*(i*inddis);

% 导入实验数据
Cr=xlsread("C:\Users\Xiangxiang\Desktop\多参改版\第二个实验数据\实验\experimentdata2.xlsx");
ra2_ex = Cr(:,1);%深度
data = Cr(:,2);%信号强度
ra2_ex = ra2_ex';
data = data';
wide_ex=max(ra2_ex);

% 一维插值处理
x1_ex=ra2_ex;
y1_ex=data;
ra2_ex=0:inddis:max(ra2_ex);
data=interp1(x1_ex,y1_ex,ra2_ex,'spline');
data(data>1)=1;
data(data<0)=0;
data=data';
fit=[data;0];
plot(ra2_ex,data);
title('Processed Experimental Data');
grid on;
DataLength = length(fit);

% 定义MRI参数定义域（保留你的设置）
DefineDomain = [0 0 0 0 0;
                3 0.001 3 1 0]; % w, lambda, sigema_0, sigema_k, sigema_c

% 定义CPSO基准参数（后续用于Morris样本生成的范围参考）
D = 5; % MRI参数维度
maxIterations = 200; % 最大迭代次数（固定，不参与灵敏度分析）
convergeThreshold = 0.01; % 收敛阈值（用于计算收敛迭代数）

%% ============== 步骤2：Morris筛选法配置（7个CPSO参数，无自定义function） ==============
% 2.1 定义待分析的CPSO参数信息（名称、下界、上界）
numSensParams = 7; % 7个CPSO参数参与灵敏度分析
sensParamNames = {'numSubSwarms', 'popSize', 'wa', 'c1', 'c2', 'coldStartFrequency', 'resetRatio'};
% 参数下界
sensParamLower = [5, 2, 0.1, 0.5, 0.5, 2, 0.2];
% 参数上界
sensParamUpper = [15, 10, 0.9, 2.5, 2.5, 10, 1.0];
% Morris实验设计参数
numTrajectories = 4; % 轨迹数（平衡计算量和结果稳定性）
numLevels = 6; % 每个参数的等间距水平数
delta = numLevels / (2 * (numLevels - 1)); % Morris设计核心参数

% 2.2 直接嵌入：生成Morris样本集（无辅助function，纯脚本逻辑）
fprintf('\nGenerating Morris sample set...\n');
morrisDesign = [];
for traj = 1:numTrajectories
    % 生成初始点
    x0 = randi([0, numLevels-2], 1, numSensParams);
    % 生成轨迹（嵌入原辅助函数逻辑）
    for p = 1:numSensParams
        x = x0;
        x(p) = x(p) + 1;
        morrisDesign = [morrisDesign; x0; x];
    end
end
morrisDesign = morrisDesign / (numLevels - 1); % 归一化处理
numSamples = size(morrisDesign, 1);

% 2.3 将Morris样本映射到CPSO参数的实际取值范围
morrisSamples = zeros(numSamples, numSensParams);
for i = 1:numSensParams
    % 归一化到[0,1]
    morrisSamples(:, i) = (morrisDesign(:, i) - min(morrisDesign(:, i))) / (max(morrisDesign(:, i)) - min(morrisDesign(:, i)));
    % 映射到实际参数范围
    morrisSamples(:, i) = sensParamLower(i) + morrisSamples(:, i) .* (sensParamUpper(i) - sensParamLower(i));
end

% 2.4 整数参数取整（numSubSwarms/popSize/coldStartFrequency）
morrisSamples(:, 1) = round(morrisSamples(:, 1)); % numSubSwarms
morrisSamples(:, 2) = round(morrisSamples(:, 2)); % popSize
morrisSamples(:, 6) = round(morrisSamples(:, 6)); % coldStartFrequency

% 2.5 初始化结果存储矩阵
sampleResults = zeros(numSamples, 3); % 每列：最终RMSE、收敛迭代数、运行耗时

%% ============== 步骤3：批量运行CPSO（遍历所有Morris样本，纯脚本） ==============
fprintf('\nStarting batch CPSO run for Morris samples (%d samples total)...\n', numSamples);
for sampleIdx = 1:numSamples
    % 提取当前样本的CPSO参数
    currentCPSOParams_numSubSwarms = morrisSamples(sampleIdx, 1);
    currentCPSOParams_popSize = morrisSamples(sampleIdx, 2);
    currentCPSOParams_wa = morrisSamples(sampleIdx, 3);
    currentCPSOParams_c1 = morrisSamples(sampleIdx, 4);
    currentCPSOParams_c2 = morrisSamples(sampleIdx, 5);
    currentCPSOParams_coldStartFrequency = morrisSamples(sampleIdx, 6);
    currentCPSOParams_resetRatio = morrisSamples(sampleIdx, 7);
    
    % 固定随机种子，消除随机性影响
    rng(123 + sampleIdx);
    
    % 启动计时
    tic();
    
    % ========== 初始化当前样本的CPSO粒子 ==========
    subSwarmPositions = zeros(currentCPSOParams_popSize, D, currentCPSOParams_numSubSwarms);
    subSwarmVelocities = zeros(currentCPSOParams_popSize, D, currentCPSOParams_numSubSwarms);
    subSwarmBestPositions = zeros(currentCPSOParams_popSize, D, currentCPSOParams_numSubSwarms);
    subSwarmBestFitness = Inf(currentCPSOParams_popSize, currentCPSOParams_numSubSwarms);
    globalBestPosition = zeros(1, D);
    globalBestFitness = Inf;
    fitnessHistory = zeros(maxIterations, 1);
    
    % 粒子初始化
    for i = 1:currentCPSOParams_numSubSwarms
        for j = 1:currentCPSOParams_popSize
            subSwarmPositions(j, :, i) = DefineDomain(1,:) + rand(1, D) .* (DefineDomain(2,:) - DefineDomain(1,:));
            subSwarmVelocities(j, :, i) = zeros(1, D);
            
            % 计算初始适应度（调用你的卷积函数）
            conv_result = convolution2(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5));
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
    
    % ========== 当前样本的CPSO迭代过程 ==========
    for iter = 1:maxIterations
        for i = 1:currentCPSOParams_numSubSwarms
            for j = 1:currentCPSOParams_popSize
                % 更新速度
                r1 = rand(1, D);
                r2 = rand(1, D);
                subSwarmVelocities(j, :, i) = currentCPSOParams_wa * subSwarmVelocities(j, :, i) + ...
                                              currentCPSOParams_c1 * r1 .* (subSwarmBestPositions(j, :, i) - subSwarmPositions(j, :, i)) + ...
                                              currentCPSOParams_c2 * r2 .* (globalBestPosition - subSwarmPositions(j, :, i));
                
                % 更新位置并限制定义域
                subSwarmPositions(j, :, i) = subSwarmPositions(j, :, i) + subSwarmVelocities(j, :, i);
                subSwarmPositions(j, :, i) = max(subSwarmPositions(j, :, i), DefineDomain(1,:));
                subSwarmPositions(j, :, i) = min(subSwarmPositions(j, :, i), DefineDomain(2,:));
                
                % 计算新适应度
                conv_result = convolution2(fit, subSwarmPositions(j, 1), subSwarmPositions(j, 2), subSwarmPositions(j, 3), subSwarmPositions(j, 4), subSwarmPositions(j, 5));
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
        
        % 冷启动（粒子重置）
        if mod(iter, currentCPSOParams_coldStartFrequency) == 0
            for i = 1:currentCPSOParams_numSubSwarms
                numReset = floor(currentCPSOParams_popSize * currentCPSOParams_resetRatio);
                for resetIdx = 1:numReset
                    idx = randi([1, currentCPSOParams_popSize]);
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

%% ============== 步骤4：计算Morris灵敏度指数（无自定义function，直接嵌入） ==============
fprintf('\nCalculating Morris sensitivity indices...\n');
% 提取核心输出指标（最终RMSE，可替换为sampleResults(:,2)收敛迭代数）
outputMetric = sampleResults(:, 1);

% 直接嵌入：计算Morris指数（主效应mu、交互效应sigma、总效应mu_star）
% 直接嵌入：计算Morris指数（主效应mu、交互效应sigma、总效应mu_star）
mu = zeros(1, numSensParams);
sigma = zeros(1, numSensParams);
mu_star = zeros(1, numSensParams);

for p = 1:numSensParams
    % 提取该参数的所有取值和对应输出
    paramValues = morrisSamples(:, p);
    [uniqueVals, ~, idx] = unique(paramValues);
    effect = []; % 初始化为空行向量（后续统一串联行向量）
    
    % 计算参数效应（修复horzcat错误：统一维度为行向量，增加判空）
    for valIdx = 1:length(uniqueVals)
        val = uniqueVals(valIdx);
        valOutput = outputMetric(idx == valIdx);
        
        % 步骤1：计算差值前，先判断valOutput长度是否≥2（避免diff返回空）
        if length(valOutput) >= 2
            % 步骤2：将diff返回的列向量转为行向量（用(:)'转置），统一维度
            valDiff = diff(valOutput); % 得到列向量
            valDiff_row = valDiff(:)'; % 转为行向量，匹配effect的维度
            
            % 步骤3：水平串联（此时均为行向量，无维度错误）
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
% 整理灵敏度结果表格
sensitivityResult = table(...
    sensParamNames', mu', sigma', mu_star', ...
    'VariableNames', {'ParamName', 'MainEffect_mu', 'InteractionEffect_sigma', 'TotalEffect_mu_star'});
% 按总效应排序（从高到低）
sensitivityResult = sortrows(sensitivityResult, 'TotalEffect_mu_star', 'descend');

% 打印灵敏度排序结果
fprintf('\n==================== CPSO Parameters Sensitivity Ranking ====================\n');
disp(sensitivityResult);

%% ============== 步骤5：灵敏度分析可视化结果（纯脚本，无依赖） ==============
% 5.1 图1：主效应vs交互效应散点图（核心可视化，修复colorbar标签报错）
figure('Position', [100, 100, 1000, 600]);
scatter(mu, sigma, 150, mu_star, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Main Effect (\mu)');
ylabel('Interaction Effect (\sigma)');
title('Morris Global Sensitivity Analysis: CPSO Parameters');
% 修复colorbar标签：先创建colorbar对象，再用ylabel添加标签（兼容所有Matlab版本）
cbar = colorbar; % 创建颜色条对象，不直接设置Label
ylabel(cbar, 'Total Effect (\mu^*)'); % 给颜色条添加y轴标签，替代只读的Label属性
grid on;
% 添加参数标签
for i = 1:numSensParams
    text(mu(i)+0.005, sigma(i)+0.005, sensParamNames{i}, 'FontSize', 10, 'FontWeight', 'bold');
end
% 5.2 图2：灵敏度总效应柱状图（排序展示）
figure('Position', [200, 200, 900, 500]);
paramLabels = sensitivityResult.ParamName;
totalEffect = sensitivityResult.TotalEffect_mu_star;
barh(1:numSensParams, totalEffect, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
set(gca, 'YTickLabel', paramLabels);
xlabel('Total Sensitivity Effect (\mu^*)');
ylabel('CPSO Parameters');
title('CPSO Parameters Sensitivity Ranking (Total Effect)');
grid on;
% 标注数值
for i = 1:numSensParams
    text(totalEffect(i)+0.001, i, sprintf('%.4f', totalEffect(i)), 'FontSize', 8, 'VerticalAlignment', 'middle');
end

% 5.3 图3：所有参数的RMSE影响趋势图（每个参数独立子图）
figure('Position', [300, 300, 1400, 600]); % 放大画布，适配7个子图
numAllParams = numSensParams; % 所有7个参数
layoutRow = 2; % 子图行数
layoutCol = 4; % 子图列数（2行4列，刚好放下7个参数，无拥挤）

for idx = 1:numAllParams
    % 提取当前参数的信息（遍历所有7个参数，不再只取前3）
    paramRank = idx; % 对应灵敏度结果的排名（已按mu*降序排序）
    paramName = sensitivityResult.ParamName{paramRank};
    paramCol = find(strcmp(sensParamNames, paramName));
    paramValues = morrisSamples(:, paramCol);
    paramRMSE = sampleResults(:, 1);
    
    % 子图布局：2行4列，依次排列
    subplot(layoutRow, layoutCol, idx);
    % 保持原有散点图风格
    scatter(paramValues, paramRMSE, 50, 'b', 'filled');
    xlabel(paramName);
    ylabel('Final RMSE');
    title(['Rank ', num2str(idx), ': ', paramName]); % 标题增加排名，对应总效应排序
    grid on;
    % 保持原有趋势线风格（1次多项式拟合，线性趋势）
    p = polyfit(paramValues, paramRMSE, 1);
    hold on;
    plot(paramValues, polyval(p, paramValues), 'r-', 'LineWidth', 1.5);
    hold off;
end

% 调整子图间距，避免标签重叠（优化论文排版效果）
sgtitle('RMSE Influence Trend of All CPSO Parameters (Sorted by Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
set(gcf, 'Color', 'w'); % 画布背景设为白色，更贴合论文

% %% ============== 步骤6：保留你原有CPSO的结果输出和对比 ==============
% % 提取当前最优参数（批量运行后的最终结果）
% w = globalBestPosition(1);
% lambda = globalBestPosition(2);
% sigema_0 = globalBestPosition(3);
% sigema_k = globalBestPosition(4);
% sigema_c = globalBestPosition(5);
% error = globalBestFitness;
% 
% % 输出参数对比
% output_str = sprintf(['\nOriginal and Optimized Parameters Comparison:\n\n' ...
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

% 结果对比图
% result = convolution2(fit, w, lambda, sigema_0, sigema_k, sigema_c);
% figure;
% x = 1:length(fit);
% plot(x, fit, '-b', 'DisplayName', 'Smoothed Input');
% hold on;
% plot(x, result, '-r', 'DisplayName', 'CPSO Fitting Result');
% plot(x, fit - result, '-g', 'DisplayName', 'Difference (Input - Result)');
% xlabel('Points');
% ylabel('Values');
% legend('Location', 'best');
% title('Comparison of Smoothed Input and CPSO Fitting Result');
% grid on;
% hold off;
% 
% % 总耗时
% fprintf('\nTotal runtime: %.2fs\n', toc());