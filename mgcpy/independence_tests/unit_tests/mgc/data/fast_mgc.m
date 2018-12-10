simulations = ["linear_sim", "exp_sim", "cub_sim", "joint_sim", "step_sim", "quad_sim", "w_sim", "spiral_sim", "ubern_sim", "log_sim", "root_sim", "sin_sim", "sin_sim_16", "square_sim", "two_parab_sim", "circle_sim", "ellipsis_sim", "square_sim_", "multi_noise_sim", "multi_indep_sim"];
for idx = 1:numel(simulations)
    sim_name = simulations(idx);
    X = csvread(strcat("data/",sim_name,"_x.csv"));
    Y = csvread(strcat("data/",sim_name,"_y.csv"));
    [a, b, c, d, e, f] = MGCFastTest(X, Y);
    [m, n] = size(c);
    csvwrite(strcat("data/fast_mgc/",sim_name,"_fast_res.csv"), [ones(m, 1)*a, ones(m, 1)*b, c, ones(m, 1)*d, ones(m, 1)*e, ones(m, 1)*f])
end