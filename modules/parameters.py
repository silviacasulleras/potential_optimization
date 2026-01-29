import numpy as np

def define_physical_params():
    N = 6  # maximum degree of the potential
    d = 1e6  # length d of the potential parametrization
    chi = 1e-3  # ratio omega/Omega
    S_Omega_exp_list = [-11,-10,-9,-8,-7,-6]   # Value of the exponent of the noise strength
    position_dependent_noise = True     # False recovers the old model with the constant Gamma_Omega
    theta_list = np.array([0,np.pi/2])  # Type of noise. Set theta=0 for fluctuations in the position of the noise and theta=pi/2 for fluctuations in the amplitude
    t_gas = 15.1* 2*np.pi*100*chi       # Typical collision time with a gas molecule in units of omega
    d_i = 0.05*d                        # Displacement from the top of the dw
    cost_func_type =  'coherence_length' # Cost function. It should be 'coherence_length' or 'cubicity_purity'
    cost_func_when = 'max'              # Time at which the cost function is evaluated. It should be 'max' or 'half'
    order_of_potential = 'four'         # The order of the potential should be four
    potential_types = ['DW','IDW']      # Should be an element (or more) in the list ['DW','IDW','Duffing', 'cubic']

    return {'N': N,
            'd': d,
            'chi': chi,
            'S_Omega_exp_list': S_Omega_exp_list,
            'position_dependent_noise': position_dependent_noise,
            't_gas': t_gas,
            'theta_list': theta_list,
            'order_of_potential' : order_of_potential,
            'potential_types': potential_types,
            'd_i':d_i,
            'cost_func_type':cost_func_type,
            'cost_func_when':cost_func_when
            }

def define_simulation_params():
    T_initial = 10 * 2 * np.pi                  # Simulation time for finding the classical trajectory period
    Nt_initial = 5000                           # Number of time steps for numerical simulationss
    dt_initial = T_initial / (Nt_initial - 1)   # Time step
    Nt_classical = 1000                         # Number of time steps for numerical simulations of T_c
    alpha_bound = 5                             # Value of alpha_b for the constraint in alpha_2(t)

    return {'Nt_initial': Nt_initial,
            'dt_initial': dt_initial,
            'T_initial': T_initial,
            'Nt_classical': Nt_classical,
            'alpha_bound': alpha_bound
            }

def define_optimization_params_quartic(potential_type):
    maxiter = 1000           # Number of iterations of the optimization algorithm
    ftol = 1e-6              # Cost function tolerance for termination of optimization algorithm
    xtol = 1e-6              # Optimization parameter tolerance for termination of optimization algorithm
    N_it = 1
    detailed_print = False   # Set to True if we want to print results during optimization

    phys_params = define_physical_params()
    d = phys_params["d"]
    d_i = phys_params["d_i"]
    cost_func_type =phys_params['cost_func_type']

    if potential_type=='DW':                # Define parameters associated to DW potential
        a_dw = -1
        b_dw = 1
        c_dw = 0
        lb = [-np.inf, a_dw, b_dw, c_dw]    # Lower bounds for d0, a, b and l parameters
        ub = [np.inf, a_dw, b_dw, c_dw]     # Upper bounds for d0, a, b and l parameters
    elif potential_type=='IDW':             # Define parameters associated to IDW potential
        a_idw = 1
        b_idw = -1
        c_idw = 0
        lb = [-np.inf, a_idw, b_idw, c_idw] # Lower bounds for d0, a, b and l
        ub = [np.inf, a_idw, b_idw, c_idw]  # Upper bounds for d0, a, b and l
    elif potential_type=='Duffing':         # Define parameters associated to Duffing potential
        a_duffing = 1
        b_duffing = 1
        c_duffing = 0
        lb = [d_i, a_duffing, b_duffing, c_duffing]  # lower bounds for d0, a, b and l
        ub = [d - d_i, a_duffing, b_duffing, c_duffing]  # upper bounds for d0, a, b and l
    else:
        print("Potential type should be 'DW', 'IDW' or 'Duffing'.")

    A = [ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]     # Matrix for implementing the linear constraint

    choose_evolution_solver = 'numba'  # Differential equations solver used for the dynamics. Choose 'numba or 'scipy'.
    numba_scipy_comparison = False     # Set to True to compare the results of both solvers
    solver_tol = 0.05                  # Tolerance for the difference between numba and scipy
    verbose = 1

    return {'N_it': N_it,
            'detailed_print': detailed_print,
            'maxiter': maxiter,
            'ftol': ftol,
            'xtol': xtol,
            'lb': lb,
            'ub': ub,
            'A': A,
            'verbose': verbose,
            'choose_evolution_solver': choose_evolution_solver,
            'solver_tol': solver_tol,
            'numba_scipy_comparison': numba_scipy_comparison,
            "cost_func_type": cost_func_type}


def define_evaluation_params_quartic(potential_type): # Define parameters for evaluating specific quartic potentials (without optimization)

    detailed_print = True  # set to True if we want to print results during optimization

    phys_params =  define_physical_params()
    d = phys_params["d"]
    d_i = phys_params["d_i"]
    cost_func_type = phys_params['cost_func_type']

    if potential_type=='DW':
        a =  -1
        b = 1
        c = 0
    # upper bounds for d0, a, b and l
    elif potential_type=='IDW':
        a = 1
        b = -1
        c = 0
    elif potential_type=='Duffing':
        a = 1
        b = 1
        c = 0
    else:
        print("Wrong potential type.")

    choose_evolution_solver = 'numba'  # choose 'numba or 'scipy'
    solver_tol = 0.05  # tolerance for the difference between numba and scipy
    verbose = 1

    return {'d_i':d_i,
            'detailed_print': detailed_print,
            'a':a,
            'b':b,
            'c':c,
            'verbose': verbose,
            'choose_evolution_solver': choose_evolution_solver,
            'solver_tol': solver_tol,
            "cost_func_type": cost_func_type}

def define_plot_params(): # Define the parameters for the plots
    d = define_physical_params()['d']
    a = -100 * d
    b = 100 * d
    Nx = 2 ** 16  # number of points in the position grid
    dx = (b - a) / Nx  # resolution in position
    xlist = a + dx * np.arange(Nx)

    cm = 1 / 2.54
    sz = (10 * cm, 6 * cm)

    clr_opt_1 = 'blue'
    clr_comp = 'tab:orange'
    clr_opt_2 = 'tab:blue'
    clr_init = 'tab:green'
    clr_dw = 'm'

    use_tex = True
    font = 'Times'
    include_plots = ['i_dw', 'i', 'd']
    save_as = 'pdf'
    alpha_opacity = 0.5

    return {'dx': dx,
            'xlist': xlist,
            'cm': cm,
            'sz': sz,
            'clr_comp': clr_comp,
            'clr_opt_1': clr_opt_1,
            'clr_opt_2': clr_opt_2,
            'clr_init': clr_init,
            'clr_dw': clr_dw,
            'alpha_opacity': alpha_opacity,
            'use_tex': use_tex,
            'font': font,
            'include_plots': include_plots,
            'save_as': save_as}

