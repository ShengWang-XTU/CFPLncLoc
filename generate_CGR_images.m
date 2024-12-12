clc
clear
tic;
[Name, Seq] = fastaread("data/seq_homo_219.fasta"); % Training set (homo)
% [Name, Seq] = fastaread("data/seq_mus_65.fasta");   % Independent test set (mus)
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
    fig = figure;
    plot(xu, yu, 'k.')
    pbaspect([1 1 1])
    set(gca, 'LooseInset', get(gca, 'TightInset'))
    set(gca, 'looseInset', [0 0 0 0]);
    set(gcf, 'color', 'w')
    axis off
    frame = getframe(fig);
    img = frame2im(frame);
    eval(['imwrite(img, "data/CGR_homo_219/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save training set (homo)
    % eval(['imwrite(img, "data/CGR_mus_65/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save independent test set (mus)
    close all
end
toc
