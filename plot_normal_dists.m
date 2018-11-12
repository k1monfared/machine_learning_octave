% Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team
mu = 0;
for i = -3:2
	sigma = 2^i;
	x = -15:0.01*sigma:15;
	y = Normal_dist(mu,sigma,x);
	subplot(2,3,i+4)
		plot(x,y,'LineWidth',2)
		title(['Normal distribution with mu = ' num2str(mu) ' and sgima = 2^{' num2str(i) '}' ]);
		xlim([min(x), max(x)]);
		hold on
		[ymax,idxmax] = max(y);
		ylim([0,4]);
		plot(x(idxmax), ymax, 'o');
		text(x(idxmax)+.1 , ymax+.2, [num2str(x(idxmax)) ', ' num2str(ymax)]);
		plot([mu, mu], [0, ymax],'k','LineWidth',2);
		s = y(find(x==sigma));
		plot([0, sigma], [s, s],'k','LineWidth',2);
		text(sigma+.2 , s, [ '\sigma'])
end