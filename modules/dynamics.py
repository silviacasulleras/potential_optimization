import time
import numpy as np
from scipy.special import factorial
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc
from colorama import Fore

from modules.parameters import define_physical_params, define_simulation_params, define_optimization_params_quartic

def classical_eqs_static(y, t, c_ns):     # Define Hamilton equations
    phys_params = define_physical_params()
    # print(phys_params)

    chi = phys_params['chi']
    N = phys_params['N']
    d = phys_params['d']

    xc, pc = y
    dydt = [pc / chi, -1 / 2 * chi * sum(
         c_ns[1:] / factorial( np.arange(1, N + 1) -1 ) * xc ** (np.arange(1, N + 1) - 1)
        / d ** (np.arange(1, N + 1) - 2))
            ]
    return dydt


def find_classical_period(c_ns):        # Find the period of the solutions to the Hamilton equations
    sim_params = define_simulation_params()
    dt_initial = sim_params['dt_initial']
    Nt_initial = sim_params['Nt_initial']
    Nt_classical = sim_params['Nt_classical']

    chi = define_physical_params()['chi']

    tlist_initial = dt_initial * np.arange(Nt_initial)

    # Solve the classical equations of motion
    y0 = [0.0, 0.0]
    classical_sol = odeint(classical_eqs_static, y0, tlist_initial, args=(c_ns,))

    # Find time when p changes sign
    pclassical = classical_sol[:, 1]
    pc_sign = np.sign(pclassical)
    signchange = ((np.roll(pc_sign, 1) - pc_sign) != 0).astype(int)
    signchange[0] = 0
    signchange[1] = 0
    indices_changes = np.where(signchange == 1)[0]

    if len(indices_changes) > 1:
        i_t_period_classical = indices_changes[0]
        T_period_classical = 2 * tlist_initial[i_t_period_classical]
    else:
        print(Fore.MAGENTA + "Period of classical trajectory not found (suspected total T too small). Returning T." + Fore.BLACK)
        T_period_classical = tlist_initial[-1]

    dt_classical = T_period_classical / (Nt_classical - 1)
    tlist = dt_classical * np.arange(Nt_classical)

    #print(f"Classical period: T_c*omega={T_period_classical}")
    #print(f"Classical period: T_c*Omega={T_period_classical/chi}")
    return {"T_p_classical": T_period_classical, 'tlist': tlist}


def compute_classical_traj(c_ns):           # Compute the classical trajectories
    phys_params = define_physical_params()
    N = phys_params['N']
    d = phys_params['d']

    sim_params = define_simulation_params()
    Nt_classical = sim_params['Nt_classical']

    classical_period = find_classical_period(c_ns)
    T_classical = classical_period['T_p_classical']
    tlist = classical_period['tlist']
    # t_span = [tlist[0], tlist[-1]]

    # Solve the classical equations of motion for t<=T_period
    y0 = [0.0, 0.0]
    classical_sol = odeint(classical_eqs_static, y0, tlist, args=(c_ns,))
    xclassical = classical_sol[:, 0]
    pclassical = classical_sol[:, 1]

    #Define function alpha(t)
    def alpha_t(xc_t, opt_params_1):
        alpha = sum(opt_params_1[2:] / factorial(np.arange(2, N + 1)) * np.arange(2, N + 1) * (
                np.arange(2, N + 1) - 1) / 2 * xc_t ** (np.arange(2, N + 1) - 2) / d ** (np.arange(2, N + 1) - 2))
        return alpha

    alpha_t_vec = np.vectorize(alpha_t)  # function
    alpha_t_vec.excluded.add(1)

    alpha_t_list = alpha_t_vec(xclassical, c_ns)

    return {"xc": xclassical,
            "pc": pclassical,
            "alpha": alpha_t_list,
            "tlist": tlist,
            'T_classical_period': T_classical,
            }

def compute_position_dep_noise( c_ns, S_Omega, theta): # Compute the decay rate of the position dependent noise

    #Calculate the time-dependent noise gamma(t) from the classical trajectories in units of omega

    phys_params = define_physical_params()
    N = phys_params['N']
    d = phys_params['d']
    chi = phys_params['chi']
    position_dependent_noise = phys_params['position_dependent_noise']

    classical_traj =  compute_classical_traj(c_ns)
    tlist = classical_traj['tlist']
    xclassical = classical_traj['xc']

    # --------> calculate position dependent displacement noise
    def U2_d1(xlist, c_ns):
        U2deriv1 = 1/2 * sum(
            c_ns[1:N + 1] / factorial(np.arange(1, N + 1) - 1) * xlist ** (np.arange(1, N + 1) - 1) / d ** (
                        np.arange(1, N + 1) - 2))
        return U2deriv1

    U2_d1_vec = np.vectorize(U2_d1)
    U2_d1_vec.excluded.add(1)
    U2_deriv1 = U2_d1_vec(xclassical, c_ns)

    def U2_d2(xlist, c_ns):
        U2deriv2 = 1/2 * sum(
            c_ns[2:N + 1] / factorial(np.arange(2, N + 1) - 2) * (xlist / d) ** (np.arange(2, N + 1)-2) )
        return U2deriv2

    U2_d2_vec = np.vectorize(U2_d2)
    U2_d2_vec.excluded.add(1)
    U2_deriv2 = U2_d2_vec(xclassical, c_ns)

    gamma_t_list = np.pi * 1 / 2 * chi**3 * (
            d**2 * S_Omega * np.cos(theta) * U2_deriv2 ** 2 + S_Omega * np.sin(theta) * U2_deriv1 ** 2)
    max_gamma_tlist =  np.amax(gamma_t_list)
    if position_dependent_noise:
        gamma_t_list = gamma_t_list
    else:
        gamma_t_list = max_gamma_tlist*np.ones(len(tlist))  # constant with maximum value of Gamma(t) if position_dependent_noise is false

    return { "tlist": tlist,
            'gamma_t': gamma_t_list,
             "U_deriv1": U2_deriv1,
             "U_deriv2": U2_deriv2
            }

def compute_third_derivative_potential( c_ns ): # Compute the third derivative of the potential

    phys_params = define_physical_params()
    N = phys_params['N']
    d = phys_params['d']
    position_dependent_noise = phys_params['position_dependent_noise']

    classical_traj =  compute_classical_traj(c_ns)
    tlist = classical_traj['tlist']
    xclassical = classical_traj['xc']

    # --------> calculate 3rd derivative of the potential
    def U2_d3(xlist, c_ns):
        Vderiv3 = 1/2 *  sum(
            c_ns[3:N + 1] / factorial(np.arange(3, N + 1) - 3) * xlist ** (np.arange(3, N + 1) - 3)
            / d ** (np.arange(3, N + 1) - 2))
        return Vderiv3

    U2_d3_vec = np.vectorize(U2_d3)
    U2_d3_vec.excluded.add(1)
    U2_deriv3= U2_d3_vec(xclassical, c_ns)

    #Return derivative of the potential evaluated at tlist

    return U2_deriv3

def eqs_motion_2nd_moments(t, y, coefficients, chi): # Define the equations of motion for the second order moments
    #Equations for the elements of the covariance matrix
    exp_x_2, exp_p_2, exp_c, lambd = y
    dydt = [2 * exp_c / chi,
            -2 * chi * coefficients['alpha'](t) * exp_c + 4 * coefficients['gamma'](t),
            exp_p_2 / chi - chi * coefficients['alpha'](t) * exp_x_2,
            4 * coefficients['gamma'](t) *exp_x_2
            ]
    return dydt

def eqs_motion_symplectic(t, y, coefficients, chi): # Define the differential equation for the symplectic matrix
    Z_xx, Z_px, Z_xp, Z_pp = y
    dydt = [Z_xp / chi,
            Z_pp / chi,
            - chi * coefficients['alpha'](t) * Z_xx,
            - chi * coefficients['alpha'](t) * Z_px
            ]
    return dydt

def find_tolerance(S_Omega):
    r_tol = 1e-10
    return r_tol

# Define functions for the numba solver
@njit
def interp_nb(xvals, x, y):
    return np.interp(xvals, x, y)


def make_lsoda_func(tlist, alpha, gamma, chi):
    @cfunc(lsoda_sig)
    def rhs(t, y, dydt, p):
         dydt[0] = 2 * y[2] / chi
         dydt[1] = -2 * chi * interp_nb(t, tlist, alpha) * y[2] + 4 * interp_nb(t, tlist, gamma)
         dydt[2] = y[1] / chi - chi * interp_nb(t, tlist, alpha) * y[0]
         dydt[3] = 4 * interp_nb(t, tlist, gamma) * y[0]
    return rhs

def make_lsoda_func_symplectic(tlist, alpha, chi):
    @cfunc(lsoda_sig)
    def rhs(t, y, dydt, p):
         dydt[0] = y[2] / chi
         dydt[1] = y[3] / chi
         dydt[2] = - y[0] * chi * interp_nb(t, tlist, alpha)
         dydt[3] = - y[1] * chi * interp_nb(t, tlist, alpha)
    return rhs

def compute_moments(S_Omega, classical_traj, gamma, choose_solver='scipy'):

    phys_params = define_physical_params()
    chi = phys_params['chi']

    tlist = classical_traj["tlist"]
    T_classical = classical_traj['T_classical_period']
    alpha = classical_traj['alpha']

    moments_0 = np.array([1.0, 1.0, 0.0, 1.0])

    if choose_solver == 'numba':  # Solve the equations of motion of the moments using numba solver
        tol_diff = find_tolerance(S_Omega)
        rhs = make_lsoda_func(tlist, alpha, gamma, chi)
        funcptr = rhs.address
        t_eval = np.linspace(tlist[0], tlist[-1], len(tlist))  # times to evaluate solution
        sol1, success = lsoda(funcptr, moments_0, t_eval, rtol=tol_diff)
        z = sol1.T

    else:  # if not specified 'numba' explicitly, solve e.o.m. using scipy
        alpha_f = interp1d(tlist, alpha)
        gamma_f = interp1d(tlist, gamma)
        coefficients = {'alpha': alpha_f, 'gamma': gamma_f}
        t_span = [tlist[0], tlist[-1]]
        tol_diff = find_tolerance(S_Omega)
        sol = solve_ivp(eqs_motion_2nd_moments, t_span, moments_0, args=(coefficients, chi), t_eval=tlist,
                        method='RK23',
                        rtol=tol_diff)
        z = sol.y

    S_xx = z[0, :].T
    S_pp = z[1, :].T
    S_xp = z[2, :].T
    S_L = z[3, :].T

    return {'tlist': tlist,
            'S_xx': S_xx,
            'S_pp': S_pp,
            'S_xp': S_xp,
            'S_L': S_L,
            'T_classical_period': T_classical}

def compute_symplectic_M(S_Omega, classical_traj, gamma, choose_solver='scipy'): # Compute the symplectic matrix

    phys_params = define_physical_params()
    chi = phys_params['chi']

    tlist = classical_traj["tlist"]
    T_classical = classical_traj['T_classical_period']
    alpha = classical_traj['alpha']

    symplectic_0 = np.array([1.0, 0.0, 0.0, 1.0])

    if choose_solver == 'numba':  # Solve the equations of motion of the moments using numba solver
        tol_diff = find_tolerance(S_Omega)
        rhs = make_lsoda_func_symplectic(tlist, alpha, chi)
        funcptr = rhs.address
        t_eval = np.linspace(tlist[0], tlist[-1], len(tlist))  # times to evaluate solution
        sol1, success = lsoda(funcptr, symplectic_0, t_eval, rtol=tol_diff)
        Z = sol1.T

    else:  # if not specified 'numba' explicitly, solve e.o.m. using scipy
        alpha_f = interp1d(tlist, alpha)
        gamma_f = interp1d(tlist, gamma)
        coefficients = {'alpha': alpha_f, 'gamma': gamma_f}
        t_span = [tlist[0], tlist[-1]]
        tol_diff = find_tolerance(S_Omega)
        sol = solve_ivp(eqs_motion_symplectic, t_span, symplectic_0, args=(coefficients, chi), t_eval=tlist,
                        method='RK23',
                        rtol=tol_diff)
        Z = sol.y

    Z_xx = Z[0, :].T
    Z_xp = Z[1, :].T
    Z_px = Z[2, :].T
    Z_pp = Z[3, :].T

    return {'tlist': tlist,
            'Z_xx': Z_xx,
            'Z_xp': Z_xp,
            'Z_px': Z_px,
            'Z_pp': Z_pp,
            'T_classical_period': T_classical}

def compute_coherence_length(moments): # Compute the coherence length
    S_xx = moments['S_xx']
    S_pp = moments['S_pp']
    S_xp = moments['S_xp']
    S_L = moments['S_L']

    # Compute coherence length
    S_phi = (S_xx + S_pp + np.sqrt( (S_xx - S_pp) ** 2 + (2 * S_xp) ** 2 ) ) / 2

    # purity = 1 / np.sqrt(S_xx * S_pp - S_xp ** 2)
    purity2 = 1 / np.sqrt(S_L)

    #coherence_length = np.sqrt(8) * purity2 * np.sqrt(S_phi)
    coherence_length = np.sqrt(8) * purity2 * np.sqrt(S_phi)

    cost_func_when = define_physical_params()['cost_func_when']
    if cost_func_when =='max':
        peak_coherence_length = np.amax(coherence_length)
    elif cost_func_when=='half':
        peak_coherence_length =  coherence_length[int(len(coherence_length)/2)]
    else:
        print("Cost_func_when should be 'max' or 'half'.")


    return {'S_phi': S_phi,
            'purity2': purity2,
            'coherence_length': coherence_length,
            'peak_coherence_length': peak_coherence_length}


def compute_cubicity(c_ns, symplectic_M):   #Compute the cubicity

    phys_params = define_physical_params()
    chi = phys_params['chi']
    cost_func_when = phys_params['cost_func_when']

    Z_xx = symplectic_M['Z_xx']
    Z_xp = symplectic_M['Z_xp']
    Z_px = symplectic_M['Z_px']
    Z_pp = symplectic_M['Z_pp']

    tlist = symplectic_M['tlist']

    # Compute cubicity
    eta = np.sqrt(Z_xx **2 + Z_xp **2)

    V_3 = compute_third_derivative_potential( c_ns )
    #
    beta = 1/2 * chi * V_3 * eta**3

    #beta = 1 / 3 * V_3 * eta ** 3

    #beta_tilde = beta.copy()
    #beta_tilde[int(len(beta)/2):] = -beta_tilde[int(len(beta)/2):]

    angle_phi = np.arctan2( Z_xp , Z_xx )

    beta_tilde_1 = beta * np.sin( angle_phi )
    beta_tilde_2 = beta * np.cos( angle_phi )

    cubicity = np.zeros(len(tlist))
    for i, t in enumerate(tlist):
        dt = tlist[1]-tlist[0]
        cubicity[i] = np.sqrt( ( sum(beta_tilde_1[0:i]) * dt ) **2 + ( sum(beta_tilde_2[0:i]) * dt ) **2 )

    if cost_func_when=='max':
        peak_cubicity = np.amax( np.abs( cubicity) )
    elif cost_func_when=='half':
        peak_cubicity =  np.abs( cubicity[int(len(cubicity)/2)] )
    else:
        print("Cost_func_when should be 'max' or 'half'.")

    wide_pot_cond = eta**3 * V_3

    return {'eta': eta,
            'beta': beta,
            'angle_phi': angle_phi,
            'cubicity': cubicity,
            'peak_cubicity': peak_cubicity,
            'wide_pot_cond' : wide_pot_cond,
    }

def compute_blurring(symplectic_M, gamma_t):  # Compute the blurring coefficient

    #phys_params = define_physical_params()
    #chi = phys_params['chi']

    Z_xx = symplectic_M['Z_xx']
    Z_xp = symplectic_M['Z_xp']

    tlist = symplectic_M['tlist']

    eta = np.sqrt(Z_xx **2 + Z_xp **2)

    # Compute blurring coefficient
    aux = 4 * gamma_t * eta**2

    blurring = np.zeros(len(tlist))
    for i, t in enumerate(tlist):
        dt = tlist[1]-tlist[0]
        blurring[i] = sum(aux[0:i])*dt

    return {'tlist':tlist, 'blurring': blurring,
    }

def compute_evolution(c_ns, S_Omega, theta, loop=False, choose_solver='scipy'): # Compute the evolution of all quantities for a given potential defined by the coefficients c_n

    start = time.time()

    cost_func_type = define_physical_params()["cost_func_type"]

    # Compute classical trajectories
    classical_traj = compute_classical_traj(c_ns)
    tlist = classical_traj['tlist']
    T_classical = classical_traj['T_classical_period']
    alpha = classical_traj["alpha"]
    xc = classical_traj['xc']
    pc = classical_traj['pc']

    pos_dep = compute_position_dep_noise(c_ns, S_Omega, theta)
    gamma_t = pos_dep["gamma_t"]
    U_deriv1 = pos_dep["U_deriv1"]
    U_deriv2 = pos_dep["U_deriv2"]

    # Compute moments
    moments = compute_moments(S_Omega, classical_traj, gamma_t, choose_solver)

    # Compute coherence length
    coherence = compute_coherence_length(moments)
    purity2 = coherence['purity2']
    purity_half_period = purity2[ int(len(purity2)/2) ]
    purity_full_period = purity2[ -1 ]

    coherence_length = coherence['coherence_length']
    peak_coherence_length = coherence['peak_coherence_length']

    # Compute cubicity
    symplectic_M = compute_symplectic_M(S_Omega, classical_traj, gamma_t, choose_solver)
    cubicity_results = compute_cubicity(c_ns, symplectic_M)
    beta = cubicity_results["beta"]
    cubicity = cubicity_results["cubicity"]
    angle_phi = cubicity_results["angle_phi"]
    eta = cubicity_results["eta"]

    #Compute weak nonlinearity condition
    condition = eta*U_deriv2/ U_deriv1

    # Compute blurring coefficient
    blurring = compute_blurring(symplectic_M,gamma_t)["blurring"]

    if cost_func_type == 'coherence_length':
        figure_of_merit = peak_coherence_length
        print(f"Figure of merit = {peak_coherence_length}")
    elif cost_func_type == 'coherence_length_purity_half':
        figure_of_merit = peak_coherence_length * purity_half_period
    elif cost_func_type == 'coherence_length_purity_full':
        figure_of_merit = peak_coherence_length * purity_full_period
    elif cost_func_type == 'cubicity_blurring':
        figure_of_merit = np.amax(np.abs(cubicity/blurring))
    elif cost_func_type == 'cubicity_purity':
        figure_of_merit = np.amax(np.abs(cubicity*purity2))
    else:
        print("Cost_func_type should be 'coherence_length' or 'coherence_length_purity_half' or 'cubicity_blurring' or 'cubicity_purity'. ")

    end = time.time()

    alpha_loop = []
    coherence_length_loop = []
    peak_coherence_length_loop = []
    purity2_loop = []

    return {"tlist": tlist,
            'T_classical_period': T_classical,
            "xc": xc,
            "pc": pc,
            "alpha" : alpha,
            "beta" : beta,
            "cubicity" : cubicity,
            "blurring" : blurring,
            "angle_phi": angle_phi,
            "alpha_loop" : alpha_loop,
            'purity2': purity2,
            'peak_CL': peak_coherence_length,
            'CL': coherence_length,
            'fig_of_merit': figure_of_merit,
            'peak_CL_loop': peak_coherence_length_loop,
            'CL_loop': coherence_length_loop,
            'purity2_loop': purity2_loop,
            'S_Omega': S_Omega,
            'gamma_t': gamma_t, #gamma_t is given in units of omega
            'c_ns': c_ns,
            'condition':condition,
            }
