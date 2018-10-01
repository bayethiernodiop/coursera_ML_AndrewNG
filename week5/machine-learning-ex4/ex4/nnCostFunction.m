function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
XwithoutBias = X; %used for BP
X=[ones(m,1) X];      
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


layer2withoutSigmoidAndBias = X * Theta1';  % to compute g'(z)
laye2WithoutBias = sigmoid(layer2withoutSigmoidAndBias); 
layer2 = [ones(m,1) laye2WithoutBias]; % 5000 x 26
output = sigmoid(layer2 * Theta2'); % 5000 x 10

%  contains  the concat of all oneHotEncoding
YoneHotEncodingOfLabels=zeros(m,num_labels) ;

%  construct one hot encoding for each label
for l=1:num_labels
labelOneHotEncoding = y==l;
YoneHotEncodingOfLabels(:,l)=labelOneHotEncoding;
end 

% compute J
JYEqual1InTheFormula = YoneHotEncodingOfLabels.*log(output);
JYEqual0InTheFormula = (1-YoneHotEncodingOfLabels) .* log(1-output);
JPerObservation = sum((JYEqual1InTheFormula+JYEqual0InTheFormula),2);
J=-sum(JPerObservation)/m; 


%  regularization
if(lambda > 0)
% we don't regularize bias weights(first column of every weight matrix)
Theta1WithoutBias = Theta1(:,2:end);
Theta2WithoutBias = Theta2(:,2:end);

regTerm = (lambda*(sum(sum(Theta1WithoutBias.^2)) + sum(sum(Theta2WithoutBias.^2))))/(2*m);
J=J+regTerm;
end
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% calculation of sigma3
sigma3 = output - YoneHotEncodingOfLabels; % prediction - real labels
sigma3 = sigma3'; % so that every line contains the errors of a unit for all observations
% calculation of sigma2
sigma2 = Theta2' * sigma3;
sigma2 = sigma2(2:end,:); % first line correspond to the bias unit which is not used for computing the gradient(not linked to any node in the previous layer)
sigma2 = sigma2 .* sigmoidGradient(layer2withoutSigmoidAndBias)';

Theta1_grad = (sigma2* X)/m;
Theta2_grad = (sigma3* layer2)/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



if(lambda >0)
Theta1_grad=[Theta1_grad(:,1) (Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end)))];
Theta2_grad=[Theta2_grad(:,1) (Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end)))];

end















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
