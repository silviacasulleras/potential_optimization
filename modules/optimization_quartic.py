import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import time
from colorama import Fore, Style

from modules.parameters import define_optimization_params_quartic, define_physical_params, define_evaluation_params_quartic
from modules.dynamics import compute_classical_traj, compute_position_dep_noise, compute_moments, compute_coherence_length, compute_symplectic_M, compute_cubicity, compute_blurring, compute_evolution, define_simulation_params
from modules.output import save_to_history

import gc

def create_cn_quartic(d0, a, b, c):
    # Write the coefficients from the expansion of the double-well potential
    d = define_physical_params()['d']
    N = define_physical_params()['N']
    c_ns = np.zeros(N+1)
    # Coefficients of DW with d0 and g
    c_ns[1] = 2*(d0/d)*(-a-b*d0**2/d**2 +c*d0/d)
    c_ns[2] = 2*( a+ 3*b*(d0**2/d**2)  -2*c*d0/d)
    c_ns[3] = -12*d0*b/d +4*c
    c_ns[4] = 12*b
    c_ns[5] = 0
    c_ns[6] = 0
    return c_ns.tolist()


def cost_func_quartic(coeffs, S_Omega, theta, potential_type): # Calculate the cost function for a quartic potential
    d0   = coeffs[0]
    a  = coeffs[1]
    b  = coeffs[2]
    c  = coeffs[3]

    c_ns = create_cn_quartic(d0, a, b, c)

    d = define_physical_params()['d']

    optimization_params = define_optimization_params_quartic(potential_type)
    detailed_print = optimization_params['detailed_print']
    choose_evolution_solver = optimization_params['choose_evolution_solver']
    cost_func_type = optimization_params["cost_func_type"]

    classical_traj = compute_classical_traj(c_ns)
    gamma_t = compute_position_dep_noise(c_ns, S_Omega, theta)["gamma_t"]

    # Compute moments
    moments = compute_moments(S_Omega, classical_traj, gamma_t, choose_solver=choose_evolution_solver)

    # Compute coherence length
    if cost_func_type == 'coherence_length':
        coherence = compute_coherence_length(moments)
        peak_coherence_length = coherence['peak_coherence_length']

        figure_of_merit = peak_coherence_length


    elif cost_func_type == 'coherence_length_purity_full':
        coherence = compute_coherence_length(moments)
        peak_coherence_length = coherence['peak_coherence_length']

        purity = coherence['purity2']
        purity_half_period = purity[ -1 ]

        figure_of_merit = peak_coherence_length * purity_half_period


    elif cost_func_type == 'cubicity_purity':
        coherence = compute_coherence_length(moments)
        purity = coherence['purity2']
        symplectic_M = compute_symplectic_M(S_Omega, classical_traj, gamma_t, choose_solver = choose_evolution_solver)
        cubicity_results = compute_cubicity(c_ns, symplectic_M)
        cubicity = cubicity_results["cubicity"]
        figure_of_merit = np.amax(np.abs(cubicity * purity))
    else:
        print("Cost_func_type should be 'coherence_length' or 'coherence_length_purity_full' or 'cubicity_purity'. ")
    if detailed_print:
        print(f"""Current d0: {d0}""")
        print(f"""Current a:  {a}""")
        print(f"""Current b:  {b}""")
        print(f"""Current cost function value : {figure_of_merit}""")

    return -figure_of_merit


def constraint_alpha_quartic(coeffs):  # Function for the constraint alpha_bound-alpha(t) > 0
    alpha_bound =  define_simulation_params()['alpha_bound']
    d0   = coeffs[0]
    a  = coeffs[1]
    b  = coeffs[2]
    c  = coeffs[3]
    c_ns = create_cn_quartic(d0, a, b, c)
    classical_traj = compute_classical_traj(c_ns)
    alpha = classical_traj["alpha"]
    max_alpha = np.amax(np.abs(alpha))
    #print(Fore.GREEN + f"""Max alpha(t)={max_alpha}""")
    return alpha_bound - max_alpha


def constraint_time_quartic(coeffs):  # Function for the constraint t_gas-t_max > 0
    t_gas = define_physical_params()['t_gas']  # omega*tgas=2*pi
    d0 = coeffs[0]
    a = coeffs[1]
    b = coeffs[2]
    c  = coeffs[3]
    c_ns = create_cn_quartic(d0, a, b, c)
    classical_traj = compute_classical_traj(c_ns)
    t_max = classical_traj["T_classical_period"]  # in the units of omega
    #print(Fore.GREEN + f"""t_max={t_max}""" + Style.RESET_ALL)
    return t_gas - t_max


def generate_random_params_quartic(potential_type): # Generate random parameters for initiating the optimization

    d = define_physical_params()['d']
    d_i = define_physical_params()["d_i"]
    t_gas = define_physical_params()['t_gas']  # omega*tgas=2*pi
    optimization_params = define_optimization_params_quartic(potential_type)
    detailed_print = optimization_params['detailed_print']
    lb = optimization_params['lb']
    ub = optimization_params['ub']
    cons_tmp_alpha = -1
    cons_tmp_time = -1
    if potential_type=='DW':
        lb_d0 =  0.1*d_i
        ub_d0 = 10*np.sqrt(2*d**2-d_i**2)
    elif potential_type=='IDW':
        lb_d0 =  0.1*d_i
        ub_d0 = 10*(d-d_i)
    # Generate the initial optimization parameters inside the bounds until the constraints are satisfied
    while True:
        d0 = np.random.uniform(low=lb_d0, high=ub_d0, size=(1,))
        a = np.random.uniform(low=lb[1], high=ub[1], size=(1,))
        b = np.random.uniform(low=lb[2], high=ub[2], size=(1,))
        c = np.random.uniform(low=lb[3], high=ub[3], size=(1,))
        coeffs = np.array( [d0, a, b, c])
        cons_tmp_alpha = constraint_alpha_quartic(coeffs)
        cons_tmp_time = constraint_time_quartic(coeffs)

        if cons_tmp_time > 0 and cons_tmp_alpha > 0:
            break

    if detailed_print:
        print(f'Constraints for alpha and t_max are satisfied: max alpha(t) = {5-cons_tmp_alpha}, t_C = {t_gas-cons_tmp_time}')
        print(constraint_alpha_quartic(coeffs))
        print(constraint_time_quartic(coeffs))

    return coeffs

def perform_optimization_quartic(S_Omega, theta, potential_type, Job_int): #Function to perform optimization in a quartic potentail, with only one optimization parameter, d0
    coeffs = generate_random_params_quartic(potential_type)
    initial_d0 = coeffs[0]
    initial_a  = coeffs[1]
    initial_b  = coeffs[2]
    initial_c  = coeffs[3]

    initial_c_ns = create_cn_quartic(initial_d0, initial_a, initial_b, initial_c)

    optimization_params = define_optimization_params_quartic(potential_type)
    maxiter = optimization_params['maxiter']
    verbose = optimization_params['verbose']
    ftol = optimization_params['ftol']
    xtol = optimization_params['xtol']
    lb   = optimization_params['lb']
    ub   = optimization_params['ub']
    A    = optimization_params['A']

    choose_evolution_solver = optimization_params['choose_evolution_solver']

    # Impose constraints: 1-|max alpha(t)|> 0 and time constraint

    constraint_bounds = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=True)
    cons_alpha = NonlinearConstraint(constraint_alpha_quartic, lb = 0, ub = np.inf, keep_feasible=True )
    cons_time  = NonlinearConstraint(constraint_time_quartic,  lb = 0, ub = np.inf, keep_feasible=True )

    cons = [constraint_bounds, cons_alpha, cons_time]

    start_1 = time.time()
    res_2 = minimize(
        cost_func_quartic,
        coeffs.flatten(),
        args=(S_Omega, theta, potential_type),
        constraints=cons,
        method='trust-constr',
        options={'verbose': verbose, 'maxiter': maxiter, 'gtol': ftol, 'xtol':xtol},
    )
    end_1 = time.time()
    optimized_d0 = res_2.x[0]
    optimized_a = res_2.x[1]
    optimized_b = res_2.x[2]
    optimized_c = res_2.x[3]

    optimized_coeffs = np.array([optimized_d0, optimized_a, optimized_b, optimized_c])
    optimized_c_ns = create_cn_quartic(optimized_d0, optimized_a, optimized_b, optimized_c)


    print('Optimization run completed.')
    print(f"""Optimized d0/x0: {optimized_d0} """)
    print(f"""Time required for the optimization run: {(end_1 - start_1) / 60} min""")

    optimized_fig_of_merit = -res_2.fun
    optimized_constraint_alpha = constraint_alpha_quartic(optimized_coeffs)
    optimized_constraint_time = constraint_time_quartic(optimized_coeffs)

    evolution_results_optimization = compute_evolution(optimized_c_ns, S_Omega, theta, loop=False, choose_solver=choose_evolution_solver)
    evolution_results_initial = compute_evolution(initial_c_ns, S_Omega, theta, loop=False, choose_solver=choose_evolution_solver)

    if potential_type == 'DW':
        if Job_int > 10000:
            print("Warning! Job_int >10000 and it may lead to saving error. ")
        save_to_history(Job_int, S_Omega, theta, optimized_c_ns, optimized_fig_of_merit, optimized_constraint_alpha,
                    optimized_constraint_time,
                    evolution_results_optimization, evolution_results_initial, order_of_potential='four',
                    coeffs=optimized_coeffs)
    elif potential_type == 'IDW':
        if Job_int > 10000:
            print("Warning! Job_int >10000 and it may lead to saving error. ")
        save_to_history(10000 + Job_int, S_Omega, theta, optimized_c_ns, optimized_fig_of_merit, optimized_constraint_alpha,
                    optimized_constraint_time,
                    evolution_results_optimization, evolution_results_initial, order_of_potential='four',
                    coeffs=optimized_coeffs)
    else:
        print("Potential type should be 'DW' or 'IDW'. ")

    del evolution_results_optimization #clear from memory
    del evolution_results_initial

    gc.collect()

    return

