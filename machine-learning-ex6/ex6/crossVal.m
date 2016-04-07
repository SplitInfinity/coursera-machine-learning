function [best_C, best_sigma] = crossVal(X, Y, Xval, Yval)

vals = [0.01 0.03 0.1 0.3 1 3 10 30];
n = length(vals);

best_C = 0;
best_sigma = 0;
best_sum = 0;

for C = 1:n
	for sigma = 1:n
		model = svmTrain(X, Y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), 1e-4, 100);
		pred = svmPredict(model, Xval);
		
		new_sum = sum(pred == Yval);
		
		if (new_sum > best_sum)
			best_C = C;
			best_sigma = sigma;
			best_sum = new_sum;
		endif
	end
end


end
