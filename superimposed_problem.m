% superimposed problem
%
% Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team
clear; clc;

mu1 = 3;
sigma1 = 1;
mu2 = 3.2;
sigma2 = 3;

% generate random numbers with normal distribution
m1 = 30; %number of samples
s1 = normrnd(mu1,sigma1,[1,m1]);

m2 = 20;
s2 = normrnd(mu2,sigma2,[1,m2]);

x = -15:.01:15;
y = Normal_dist(mu1,sigma1,x);

z = Normal_dist(mu2,sigma2,x);

hold on
plot(x,y,'b','LineWidth',2);
plot(x,z,'r','LineWidth',2);
plot(s1,zeros(size(s1)),'bo','MarkerSize',10,'LineWidth',2);
plot(s2,zeros(size(s2)),'rx','MarkerSize',10,'LineWidth',2);
