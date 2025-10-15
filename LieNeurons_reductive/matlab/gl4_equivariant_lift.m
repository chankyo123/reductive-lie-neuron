clc;
clear;

syms v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16
assumeAlso([v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16], 'real')

syms h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16
assumeAlso([h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16], 'real')

%% GL(4) Basis Matrices

E = cell(1, 16);

% Diagonal generators
for i = 1:4
    E{i} = zeros(4);
    E{i}(i, i) = 1;
end

% Off-diagonal generators
idx = 5;
for i = 1:4
    for j = 1:4
        if i ~= j
            E{idx} = zeros(4);
            E{idx}(i, j) = 1;
            idx = idx + 1;
        end
    end
end

% Vectorized basis matrices
E_vec = [];
for i = 1:16
    E_vec = [E_vec, reshape(E{i}, [], 1)];
end

% Combine into Lie algebra element
x_hat = 0;
for i = 1:16
    x_hat = x_hat + eval(['v' num2str(i)]) * E{i};
end

%% Find Ad(H) for H ∈ GL(4)
H = [h1, h2, h3, h4; 
     h5, h6, h7, h8; 
     h9, h10, h11, h12; 
     h13, h14, h15, h16];

Ad_H_hat = H * x_hat * inv(H);
Ad_H_hat_vec = reshape(Ad_H_hat, [], 1);

% Solve least squares to obtain x
x = inv(E_vec' * E_vec) * (E_vec') * Ad_H_hat_vec;
var = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16];
[Ad_H_sym, b] = equationsToMatrix(x, var);

% Symbolic function
syms f(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16);
f(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16) = Ad_H_sym;

%% Find Ad(Ei)
Ad_E = {};
dAd_E = {};
for i = 1:16
    % Find Ad_E_i
    H = expm(E{i}); % Numerical exponential
    Ad_E{i} = double(f(H(1,1), H(1,2), H(1,3), H(1,4), ...
                        H(2,1), H(2,2), H(2,3), H(2,4), ...
                        H(3,1), H(3,2), H(3,3), H(3,4), ...
                        H(4,1), H(4,2), H(4,3), H(4,4)));
    dAd_E{i} = logm(Ad_E{i});
end

%% Construct dro_4
dro_4 = {};
C = [];
for i = 1:16
   dro_4{i} = kron(-E{i}', eye(16)) + kron(eye(4), dAd_E{i});
   C = [C; dro_4{i}];
end

%% Solve for the null space
[U, S, V] = svd(C);
Q = null(C);

% %% Validation Test
% H = expm(hat_gl4(rand(16, 1)));  % Random GL(4) matrix
% Ad_test = double(f(H(1,1), H(1,2), H(1,3), H(1,4), ...
%                     H(2,1), H(2,2), H(2,3), H(2,4), ...
%                     H(3,1), H(3,2), H(3,3), H(3,4), ...
%                     H(4,1), H(4,2), H(4,3), H(4,4)));
% 
% % Random perturbation vector
% w = Q * Q' * ones(64, 1) * 10;
% 
% % Reshape into 16x4 matrix
% W = reshape(w, [16, 4]);
% 
% % Test the action
% v_test = [2, 3, 1, 4]';
% H_v_test = H * v_test;
% 
% x_test = W * v_test;
% x_H_test = W * H_v_test;
% x_ad_test = Ad_test * x_test;
% 
% disp([x_H_test, x_ad_test])

%% Check rank
rank(C, 1e-10)
