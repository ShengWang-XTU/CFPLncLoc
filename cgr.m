
function [x,y] = cgr(seq)
    len = length(seq);
    x(1) = 0.5; y(1) = 0.5;
    for j = 1:len
        a = seq(j);
        switch a
        case 'A'
                v = [0,0];
        case 'C'
                v = [0,1];
        case 'G'
                v = [1,0];
        case 'T'
                v = [1,1];
        end
        x(j+1) = 0.5 * (x(j) + v(1));
        y(j+1) = 0.5 * (y(j) + v(2)); 
    end
    x(1) = [];
    y(1) = [];
end
