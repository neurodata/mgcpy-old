function GenerateSimulatedData()
    % number of simulations
    num_dims = 1;
    num_rep = 1000;
    mkdir('data');

    num_sim = 20;
    num_samples = 100;
    for type = 1:num_sim
        for n = linspace(5, num_samples, num_samples/5)
            x_mtx = zeros(num_rep, n, num_dims);
            y_mtx = zeros(num_rep, n, num_dims);
            for rep = 1:num_rep
                [x, y] = CorrSampleGenerator(type, n, num_dims, 1, 0);
                x_mtx(rep, :, :) = x;
                y_mtx(rep, :, :) = y;
            end
            X_name = strcat('data/', 'type_', num2str(type), '_size_', num2str(n), '_X.mat');
            Y_name = strcat('data/', 'type_', num2str(type), '_size_', num2str(n), '_Y.mat');
            save(X_name, 'x_mtx');
            save(Y_name, 'y_mtx');
            disp(size(x_mtx))
        end
    end
end
