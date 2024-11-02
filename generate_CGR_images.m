clc
clear
tic;
Seq = importdata("data/seq_729.csv"); % Training set for Dataset I
% Seq = importdata("data/seq_holdout_82.csv");  % Holdout test set for Dataset I
% [Name, Seq] = fastaread("data/seq_homo_219.fasta"); % Training set (homo) for Dataset II
% [Name, Seq] = fastaread("data/seq_mus_65.fasta");   % Independent test set (mus) for Dataset II
lenstd = 256;
for ind = 1:length(Seq)
    disp(ind)
    data = Seq{ind};
    len(ind) = length(data);
    lenrat = len(ind)/lenstd;
    if lenrat >= 1 && lenrat < 4
        [xu, yu] = cgr(data);
    elseif lenrat >= 4 && lenrat < 16
        [x, y] = cgr(data);
        xu = []; yu = [];
        for i = 1:length(x)
            if x(i) <= 1/2 && y(i) <= 1/2
                xu = [xu x(i)*2];
                yu = [yu y(i)*2];
            else
                xu = xu;
                yu = yu;
            end
        end
    elseif lenrat >= 16 && lenrat < 64
        [x, y] = cgr(data);
        xu = []; yu = [];
        for i = 1:length(x)
            if x(i) <= 1/4 && y(i) <= 1/4
                xu = [xu x(i)*4];
                yu = [yu y(i)*4];
            else
                xu = xu;
                yu = yu;
            end
        end
    elseif lenrat >= 64 && lenrat < 256
        [x, y] = cgr(data);
        xu = []; yu = [];
        for i = 1:length(x)
            if x(i) <= 1/8 && y(i) <= 1/8
                xu = [xu x(i)*8];
                yu = [yu y(i)*8];
            else
                xu = xu;
                yu = yu;
            end
        end
    else
        [xu, yu] = cgr(data);
    end
    fig = figure; % 新建一个figure，并将图像句柄保存到fig
    plot(xu, yu, 'k.')
    pbaspect([1 1 1])   % 将x，y，z绘图方向的绘图框长度比例改为1:1:1
    % 完全去除间隔, 可能会去除掉边界的一些信息, 请检查后使用
    set(gca, 'LooseInset', get(gca, 'TightInset'))
    % 宽度方向空白区域0， 高度方向空白区域0
    set(gca, 'looseInset', [0 0 0 0]);
    set(gcf, 'color', 'w')
    axis off
    frame = getframe(fig); % 获取frame
    img = frame2im(frame); % 将frame变换成imwrite函数可以识别的格式
    eval(['imwrite(img, "data/CGR_729/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save training set for Dataset I
    % eval(['imwrite(img, "data/CGR_holdout_82/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save holdout test set fot Dataset I
    % eval(['imwrite(img, "data/CGR_homo_219/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save training set (homo) for Dataset II
    % eval(['imwrite(img, "data/CGR_mus_65/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save independent test set (mus) fot Dataset II
    close all
end
toc