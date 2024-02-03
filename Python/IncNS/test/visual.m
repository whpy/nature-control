clear;
clc;
close all;

load("x.csv");
load("y.csv");

ra = load("../init/w_init.csv");
visual(ra,1,x,y)

function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end