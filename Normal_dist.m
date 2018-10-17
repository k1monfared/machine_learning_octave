#normal (Gaussian) distribution

function y = Normal_dist(mu,sigma,x)
	if nargin < 3
		x = -10:0.01:10;
		if nargin < 2
			mu = 0;
			if nargin < 1
				sigma = 1;
			end
		end
	end

	y = exp(- (x - mu).^2 / (2*sigma^2)) / (sqrt(2*pi) * sigma);
end