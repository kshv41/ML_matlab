function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).



r=1; c=1;
for r=1:size(z,1)
   for c=1:size(z,2)
       g(r,c)=1/(1+exp((-1)*z(r,c)));
   end
end 

% =============================================================

end
