
clc; clear; close all;
h = 0.1;
x = 0:h:1;

% 各方法數值解
y_shooting = shooting_method(x);
y_fd = finite_difference(x);
y_var = variational_method(x);

% 畫圖
plot(x, y_shooting, 'b-o', x, y_fd, 'r-s', x, y_var, 'g-^');
legend('Shooting', 'Finite Difference', 'Variational');
xlabel('x'); ylabel('y(x)');
title('Comparison of Methods');
grid on;

% 表格輸出
fprintf('\n  x     y_shooting   y_fd         y_variational   \n');
fprintf('--------------------------------------------------------\n');
for i = 1:length(x)
    fprintf('%.1f     %.6f     %.6f     %.6f      \n', ...
        x(i), y_shooting(i), y_fd(i), y_var(i));
end

function y = shooting_method(x)
    % Problem setup
    y0 = 1; y1_target = 2;
    N = length(x) - 1;
    a = x(1); b = x(end);
    h = (b - a)/N;

    % Coefficient functions
    p = @(x) -(x + 1);
    q = @(x) 2;
    r = @(x) (1 - x.^2) .* exp(-x);

    % Initialize u and v
    u1 = zeros(1, N+1); u2 = zeros(1, N+1);
    v1 = zeros(1, N+1); v2 = zeros(1, N+1);
    u1(1) = y0; u2(1) = 0;
    v1(1) = 0;  v2(1) = 1;

    for i = 1:N
        xi = x(i);

        % ---- u part ----
        k1_1 = h * u2(i);
        k1_2 = h * (p(xi)*u2(i) + q(xi)*u1(i) + r(xi));

        k2_1 = h * (u2(i) + 0.5 * k1_2);
        k2_2 = h * (p(xi + 0.5*h)*(u2(i) + 0.5*k1_2) + ...
                    q(xi + 0.5*h)*(u1(i) + 0.5*k1_1) + ...
                    r(xi + 0.5*h));

        k3_1 = h * (u2(i) + 0.5 * k2_2);
        k3_2 = h * (p(xi + 0.5*h)*(u2(i) + 0.5*k2_2) + ...
                    q(xi + 0.5*h)*(u1(i) + 0.5*k2_1) + ...
                    r(xi + 0.5*h));

        k4_1 = h * (u2(i) + k3_2);
        k4_2 = h * (p(xi + h)*(u2(i) + k3_2) + ...
                    q(xi + h)*(u1(i) + k3_1) + ...
                    r(xi + h));

        u1(i+1) = u1(i) + (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)/6;
        u2(i+1) = u2(i) + (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)/6;

        % ---- v part ----
        kk1_1 = h * v2(i);
        kk1_2 = h * (p(xi)*v2(i) + q(xi)*v1(i));

        kk2_1 = h * (v2(i) + 0.5 * kk1_2);
        kk2_2 = h * (p(xi + 0.5*h)*(v2(i) + 0.5*kk1_2) + ...
                     q(xi + 0.5*h)*(v1(i) + 0.5*kk1_1));

        kk3_1 = h * (v2(i) + 0.5 * kk2_2);
        kk3_2 = h * (p(xi + 0.5*h)*(v2(i) + 0.5*kk2_2) + ...
                     q(xi + 0.5*h)*(v1(i) + 0.5*kk2_1));

        kk4_1 = h * (v2(i) + kk3_2);
        kk4_2 = h * (p(xi + h)*(v2(i) + kk3_2) + ...
                     q(xi + h)*(v1(i) + kk3_1));

        v1(i+1) = v1(i) + (kk1_1 + 2*kk2_1 + 2*kk3_1 + kk4_1)/6;
        v2(i+1) = v2(i) + (kk1_2 + 2*kk2_2 + 2*kk3_2 + kk4_2)/6;
    end

    % Solve for w2(0)
    w2_0 = (y1_target - u1(end)) / v1(end);

    % Final solution: w = u + w2_0 * v
    y = u1 + w2_0 * v1;
end


function y = finite_difference(x)
    % Problem setup
    a = x(1);
    b = x(end);
    N = length(x) - 2;           % Number of interior points
    h = (b - a) / (N + 1);
    alpha = 1;                 % y(a)
    beta = 2;                   % y(b)

    % Define coefficient functions
    p = @(x) -(x + 1);                         % Coefficient of y'
    q = @(x) 2;                                % Coefficient of y
    r = @(x) (1 - x^2)*exp(-x);                % Right-hand side function

    % Initialize coefficient arrays
    a_ = zeros(1,N); b_ = zeros(1,N);
    c_ = zeros(1,N); d_ = zeros(1,N);

    % Step 1: i = 1
    xi = a + h;
    a_(1) = 2 + h^2 * q(xi);
    b_(1) = -1 + (h/2) * p(xi);
    d_(1) = -h^2 * r(xi) + (1 + (h/2) * p(xi)) * alpha;

    % Step 2: i = 2 to N-1
    for i = 2:N-1
        xi = a + i*h;
        a_(i) = 2 + h^2 * q(xi);
        b_(i) = -1 + (h/2) * p(xi);
        c_(i) = -1 - (h/2) * p(xi);
        d_(i) = -h^2 * r(xi);
    end

    % Step 3: i = N
    xi = b - h;
    a_(N) = 2 + h^2 * q(xi);
    c_(N) = -1 - (h/2) * p(xi);
    d_(N) = -h^2 * r(xi) + (1 - (h/2) * p(xi)) * beta;

    % Step 4: Thomas algorithm forward sweep
    l = zeros(1,N); u = zeros(1,N); z = zeros(1,N);
    l(1) = a_(1);
    u(1) = b_(1) / l(1);
    z(1) = d_(1) / l(1);

    for i = 2:N-1
        l(i) = a_(i) - c_(i) * u(i-1);
        u(i) = b_(i) / l(i);
        z(i) = (d_(i) - c_(i) * z(i-1)) / l(i);
    end

    l(N) = a_(N) - c_(N) * u(N-1);
    z(N) = (d_(N) - c_(N) * z(N-1)) / l(N);

    % Step 7: Back substitution
    w = zeros(N+2, 1);
    w(1) = alpha;
    w(end) = beta;
    w(N+1) = z(N);

    for i = N-1:-1:1
        w(i+1) = z(i) - u(i) * w(i+2);
    end

    % Output y(x)
    y = w;
end


function y = variational_method(x)
    % Parameters
    l = 1;                     % Interval [0, l]
    a = 1; b = 2;             % Boundary conditions
    n = 5;                     % Number of basis functions
    h = x(2) - x(1);

    % Given functions
    p = @(x) 1;               % p(x)
    dpdx = @(x) 0;            % p'(x)
    q = @(x) 2;               % q(x)
    f = @(x) (1 - x.^2).*exp(-x);  % f(x)

    % Homogeneous part: y(x) = y1(x) + y2(x)
    y1 = a*(1 - x/l) + b*(x/l);
    F = f(x) + (b - a)/l * dpdx(x) - q(x).*y1;

    % Basis functions: φ_i(x) = sin(iπx)
    phi  = @(i, x) sin(i * pi * x / l);
    dphi = @(i, x) (i * pi / l) * cos(i * pi * x / l);

    % Assemble A and b
    A = zeros(n);
    b_vec = zeros(n, 1);

    for i = 1:n
        for j = 1:n
            integrand_A = p(x).*dphi(i, x).*dphi(j, x) + q(x).*phi(i, x).*phi(j, x);
            A(i,j) = trapz(x, integrand_A);
        end
        integrand_b = F .* phi(i, x);
        b_vec(i) = trapz(x, integrand_b);
    end

    % Solve for c
    c = A \ b_vec;

    % Reconstruct y(x)
    y2 = zeros(size(x));
    for i = 1:n
        y2 = y2 + c(i) * phi(i, x);
    end

    y = y1 + y2;
end
