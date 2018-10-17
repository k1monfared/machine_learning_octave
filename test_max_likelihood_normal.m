% parameter estimation test
% true parameters
mu = 3;
sigma = 2;
% generate random numbers with normal distribution
m = 20; #number of samples
s = normrnd(mu,sigma,[1,m]);
% maximum likelihood parametr fits
mu_est = mean(s);
sigma_est = std(s);
% plot to see how well we did
x = min(s)-.1*sigma_est:0.01*sigma_est:max(s)+.1*sigma_est;
y = Normal_dist(mu_est,sigma_est,x);
z = Normal_dist(mu,sigma,x);

plot(x,y,'LineWidth',2,'k');
title(['mu error = ' num2str(mu_est-mu) ', sgima error = ' num2str(sigma_est - sigma)],'interpreter','none');
xlim([min(s)-.1*sigma_est,max(s)+.1*sigma_est]);
hold on
plot(x,z,'LineWidth',2,'g');
legend('estimate','true');

plot(s,zeros(size(s)),'rx','MarkerSize',10,'LineWidth',2);