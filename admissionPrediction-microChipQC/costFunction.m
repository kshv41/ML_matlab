function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
    z=X*theta;   %z-(mX1 matrix)
    h=sigmoid(z);
    k=log(h);
    l=log(1-h); 
%------- Loop implementation---------------
%for i=1:m
%  J=J+((-1)*y(i)*k(i))-((1-y(i))*l(i));
  
%end
%J=J/m;
%-------- Vector implementation------------
J= (1/m)* ((k'*(-y))-(l'*(1-y)));

%v=1;
%u=1;
%------- Loop implementation---------------
%for v=1:size(X,2)
%   for u=1:size(X,1)
%    grad(v)= grad(v)+((h(u)-y(u))*X(u,v));
%   end
%end

%grad=grad/m;
%-------- Vector implementation------------
grad = (1/m)* X'*(h-y); 

  
     







% =============================================================

end
