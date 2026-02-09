function [outputArg1] = convolution(~,w,lambda,sigema_0,sigema_k,sigema_c)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
inddis=0.5;
% 线性
%  sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0+sigema_k*(i*inddis);
%常数
 sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_0;
%开根号
 %sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sqrt(sigema_0+sigema_k*(i*inddis)^2);
%指数
%  sigma_change=@(sigema_0,sigema_k,i,inddis,sigema_c)sigema_k*sigema_0^(sigema_c*(i*inddis));
%构造膜层结构
% obj_tn=[16 14.5 15 15 15 15 15 15];
% obj_tn=[obj_tn 10.388];
% obj_tn=[16 14.5 15 15 15 15 15 15];
% obj_tn=[obj_tn 10.388];
obj_tn = 10*ones(1,6);%每层"xx"nm*（1，xx层）
% obj_tn=[16 14.5 15 15 15 15 15 15];
obj_tn=[obj_tn 15];
cons=5;


%% 卷积  
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
result=c*depth_data*inddis;%c*depth=∫gX,缺dz，所以result=c*depth_data*inddis
result(result>1)=1;
result(result<0)=0;
fit=result;   %fit=拟合结果
inintA=result;   %i=insert n=noise
outputArg1=inintA;
end


