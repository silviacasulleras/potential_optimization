import modules.parameters as par
import modules.optimization_quartic as opt_quartic
import modules.output as out
import modules.plots as pl
import numpy as np
from multiprocessing import Pool

Job_int = 1                     # Index of the optimization run
#Job_int = int(sys.argv[1])     # Uncomment to obtain the value of Job_int passed from the terminal
n_cores = 1                     # Number of cores to be used to perform the optimization runs for different values of noise strength in parallel

def f(S_Omega_exponent):
    S_Omega = 10 ** (S_Omega_exponent)
    theta_list = par.define_physical_params()["theta_list"]                   # Values of theta considered, which describe the type of noise
    potential_types = par.define_physical_params()["potential_types"]         # Types of potentials considered in the optimization
    for theta in theta_list:
        print('================================================================================')
        print(f'Optimization for S/Omega=1e{S_Omega_exponent} and theta/pi = {theta/np.pi}')
        print(f"""Optimization iteration: i = {Job_int}""")
        print('================================================================================')
        for potential_type in potential_types:
            opt_quartic.perform_optimization_quartic(S_Omega, theta, potential_type, Job_int)   # Perform the optimization for each noise strength

if __name__ == '__main__':
    with Pool(n_cores) as p:
        S_Omega_exponent_list = par.define_physical_params()["S_Omega_exp_list"]
        theta_list = par.define_physical_params()["theta_list"]                # Values of theta considered, which describe the type of noise
        p.map(f, S_Omega_exponent_list)                                        # Perform the optimization for all noise strengths
        out.save_best_run()                                                    # Save the results of the best optimization run in the 'history' folder
        pl.plot_fig_of_merit_vs_S(logscale=True,  optimization_finished=True)  # Plot the results for the optimal potential
