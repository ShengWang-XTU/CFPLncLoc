clc
clear
tic;
% [Name, Seq] = fastaread("seq_homo_219.fasta");  % Training set (homo)
[Name, Seq] = fastaread("seq_test.fasta");  % Independent test set (mus)
lenstd = 256;  % 2^p, p=8, The image update parameter p corresponds to the number of new image scatter points
for ind = 1:length(Seq)
    disp(ind)
    data = Seq{ind};
    len(ind) = length(data);
    lenrat = len(ind)/lenstd;  % The ratio of the original image (the number of scatter points) to the updated image (the number of scatter points)
    if lenrat >= 1 && lenrat < 4
        [xu, yu] = cgr(data);  % When 1 < ratio < 4, the original image is not updated.
    elseif lenrat >= 4 && lenrat < 16
        [x, y] = cgr(data);
        xu = []; yu = [];
        for i = 1:length(x)
            if x(i) <= 1/2 && y(i) <= 1/2  % When 4 < ratio < 16, the coordinates of the image are retained only 1/2 near the origin.
                xu = [xu x(i)*2];  % Then, the coordinates of the image zoom 2 times.
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
    pbaspect([1 1 1])  % Sets the frame aspect ratio to 1:1:1
    set(gca, 'LooseInset', get(gca, 'TightInset')) % Remove white spaces
    set(gca, 'looseInset', [0 0 0 0]);  % Completely remove the interval
    set(gcf, 'color', 'w')  % Sets the background color to white
    axis off
    frame = getframe(fig);
    img = frame2im(frame);
    % eval(['imwrite(img, "CGR_homo_219/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save training set (homo)
    eval(['imwrite(img, "CGR_test/CGR_', num2str(lenstd),'_', num2str(ind), '.png");']); % save independent test set (mus)
    close all
end
toc
