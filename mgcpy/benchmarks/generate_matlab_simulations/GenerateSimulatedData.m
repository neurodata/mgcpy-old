function GenerateSimulatedData()
    % number of simulations
    num_dims = 1;
    num_rep = 1000;

    % noise values determined emphircally, to show that MGC is superior
    sim_types = [1 2 3 4 5 6 7 8 9   10 11   12   13  14   15 16   17 18 19 20];
    noises =    [1 2 0 0 4 1 3 0 1.7  1  6  1.5  2.2   0  0.3  1  2.8  1  0  0];
    noise_map = containers.Map(sim_types, noises);

    num_sims = 1:20;
    num_samples = 100;

    for type = num_sims
        for n = linspace(5, num_samples, num_samples/5)
            x_mtx = zeros(num_rep, n, num_dims);
            y_mtx = zeros(num_rep, n, num_dims);
            for rep = 1:num_rep
                [x, y] = CorrSampleGenerator(type, n, num_dims, 1, noise_map(type));
                x_mtx(rep, :, :) = x;
                y_mtx(rep, :, :) = y;
            end
            base_path = '../hypothesis_tests/two_sample_test/sample_data_power_sample_sizes/';
            X_name = strcat(base_path, 'type_', num2str(type), '_size_', num2str(n), '_X.mat');
            Y_name = strcat(base_path, 'type_', num2str(type), '_size_', num2str(n), '_Y.mat');
            save(X_name, 'x_mtx');
            save(Y_name, 'y_mtx');
            disp(size(x_mtx))
        end
    end
end
