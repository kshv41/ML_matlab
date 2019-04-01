function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z=X*theta;   %z-(mX1 matrix)
    h=sigmoid(z);
    k=log(h);
    l=log(1-h); 
    f=0;
%    d=0; 
%   for i=1:m
%      f=f+((-1)*y(i)*k(i))-((1-y(i))*l(i));
%   end
   
% w=theta.^2;
% c=sum(w)-w(1);
% d=c*lambda/2;

f= (1/m)* ((k'*(-y))-(l'*(1-y)));

reg = (lambda/(2*m) )* sum(theta(2:n).^2);
 

J=f+reg;







%for q=1:n
% a=(h-y);
% b=a.*X(:,q);
% grad(q)=sum(b)/m;
%end


%for r=2:n
% grad(r)=grad(r)+(lambda/m)*theta(r);
%end

theta_reg = theta;
theta_reg(1) = 0;
grad = (1/m)* X'*(h-y) + (lambda/m * theta_reg) ;




% =============================================================

end
