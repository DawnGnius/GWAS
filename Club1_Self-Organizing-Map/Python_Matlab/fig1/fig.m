clear;clc;
filename = 'fig.gif';  
% Iters = [1:9 10*(1:9) 100*(1:9) 1000*(1:9) 10000*(1:9) 100000*(1:10)];
for i = 0:36
    str = [cd '\fig' num2str(i) '.png']; 
    Img = imread(str);
    % Img = imresize(Img, [600, 800]);
    imshow(Img);
    set(gcf, 'visible', 'off');  
    q = get(gca,'position');
    q(1) = 0;
    q(2) = 0;
    set(gca, 'position',q);
    frame = getframe(gcf, [0, 0, 800, 600]);%
    im = frame2im(frame); 
    imshow(im);
    [I, map] = rgb2ind(im, 256);
    if i == 1
        imwrite(I, map, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    else
        imwrite(I, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
    end
end
