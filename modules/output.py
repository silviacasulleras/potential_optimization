import os
import numpy as np
import scipy
from colorama import Fore

from modules.parameters import define_physical_params, define_optimization_params_quartic
from modules.dynamics import find_classical_period

def define_filepaths(S_Omega, theta, order_of_potential):
    """Defines the paths to the .txt file with the cn_s, and the folders with all the saved .npy data."""

    phys_params = define_physical_params()
    N = phys_params['N']
    d = phys_params['d']
    d_i = phys_params['d_i']
    d_exponent = int(np.log10(d))
    S_Omega_exponent = int( np.log10(S_Omega))
    theta_div_pi = theta/np.pi

    common_path = f'd=1e{d_exponent}_Gamma=1e{S_Omega_exponent}_theta={theta_div_pi:.2f}pi'  # common part for all paths (sextic potential)
    common_title_2 = f'$d/X_\\Omega$=1e{d_exponent}, $S \\Omega $=1e{S_Omega_exponent}, $\\theta/\\pi={theta_div_pi:.2f}$, $d_i/d={d_i/d}$'  # common for all titles


    if order_of_potential == 'four':
        path_to_txt = 'history/' + common_path + '_optimization_res.txt'
        path_to_evo_folders = 'history/evolution.out/' + common_path
        path_to_plot_folders = 'plots/' + common_path
        fig_title = f'Optimization results for sextic potential, ' + common_title_2
        fig_ns = f'Comparing numba and scipy for double well ($N$={N}, ' + common_title_2 + ') '

        path_to_txt_final = 'history/' + common_path + '_optimization_res_final.txt'
        path_to_evo_folders_final = 'history/evolution_final.out/' + common_path
    else:
        print("Wrong potential order")


    return {
        'path_to_txt': path_to_txt,
        'path_to_evo_folders': path_to_evo_folders,
        'path_to_plot_folders': path_to_plot_folders,
        'fig_title': fig_title,
        'fig_title_numba_scipy': fig_ns,
        'path_to_txt_final': path_to_txt_final,
        'path_to_evo_folders_final': path_to_evo_folders_final
    }


def create_folders():
    # Creates folders for saving the optimization results
    if not os.path.exists(f'history'):
        os.mkdir(f'history')

    if not os.path.exists(f'plots'):
        os.mkdir(f'plots')

    if not os.path.exists(f'history/evolution.out'):
        os.mkdir(f'history/evolution.out')

    if not os.path.exists(f'history/evolution_final.out'):
        os.mkdir(f'history/evolution_final.out')


def create_subfolders(Gamma_Omega, theta, order_of_potential):
    create_folders()
    filepaths = define_filepaths(Gamma_Omega, theta, order_of_potential)
    path_to_evo_folders = filepaths['path_to_evo_folders']
    path_to_evo_folders_final = filepaths['path_to_evo_folders_final']
    if not os.path.exists(path_to_evo_folders):
        os.mkdir(path_to_evo_folders)
    if not os.path.exists(path_to_evo_folders_final):
        os.mkdir(path_to_evo_folders_final)

def add_header(file, path, order_of_potential):
    """Function to add header to the history .txt file if it is empty."""
    phys_params = define_physical_params()
    N = phys_params['N']
    header = "#"
    if order_of_potential == 'four':
        header +=  "Job_int" + "\t" "d0/x0" + "\t"  +  "a" + "\t"  +  "b" + "\t"  +  "c" + "\t"
        for l in range(5):
            header += f"""c{l}""" + "\t "
        header += "Class. period" + "\t"
        header += "Fig. of merit" + "\n"
        if os.stat(path).st_size == 0:
            file.write(header)
    else:
        print("Potential order should be 'four'.")

def add_results(file, Job_int, c_ns, fig_of_merit, constr_a, constr_t, order_of_potential, classical_period, coeffs=[]):
    """Function to add results to the history .txt file."""
    if constr_a < -0.1 or constr_t < -0.1:
        print(Fore.RED + f'Optimized parameters do not satisfy the constraints. Not saved to the history folder.' + Fore.BLACK)
        return

    # the function will proceed to this point only if alpha constraint is satisfied
    print(f'The optimized parameters satisfy the constraints. Saved to the history folder.')

    phys_params = define_physical_params()
    N = phys_params['N']
    # create the string with the results
    results_string = str(Job_int) + "\t "
    if order_of_potential == 'four':  # data to add in case of six order potential optimization
        for m in range(len(coeffs)):
            results_string += str(coeffs[m]) + "\t"
        for l in range(5):
            results_string += str(c_ns[l]) + "\t "
    else:                          # data to add in case of normal optimization
        for l in range(N + 1):
            results_string += str(c_ns[l]) + "\t "
    results_string += str(classical_period) + "\t" +  str(fig_of_merit)  + "\n"
    file.write(results_string)


def save_npy_data(results, keys, path, name, count):
    """Save data from results specified by the list of keys, to the filepath specified by path, name and count.
    results = evolution results ;; keys = list of parameters from evolution that you want to save. """
    for key in keys:
        np.save(path + '/' + name + '_' + key + '_' + f'{count:06d}.npy', results[key])

def load_npy_data(keys, path, name, count):
    """Load data from history specified by the list of keys."""

    load_results = {}

    for key in keys:
        load_results[key] = np.load(path + '/' + name + '_' + key + '_' + f'{count:06d}.npy')

    return load_results


def save_to_history(Job_int, S_Omega, theta, c_ns, fig_of_merit, constraint_alpha, constraint_time,
                    evo_results_optim, evo_results_init, order_of_potential, numba_evolution=False, coeffs=[], optimization_finished=False):
    """Save optimal c_ns to the .txt file in the history folder. Save evolution data from the optimization run as a
    .npy file to the folder history/evolution.out"""

    position_dependent_noise = define_physical_params()['position_dependent_noise']
    create_subfolders(S_Omega, theta, order_of_potential)  # create all required folders if they
    filepaths = define_filepaths(S_Omega, theta, order_of_potential)

    if optimization_finished == True:
        path_to_txt = filepaths['path_to_txt_final']
        path_to_evo_folders = filepaths['path_to_evo_folders_final']
        file = open(path_to_txt, 'w')  # open file in write mode (overwrites previous files)
    else:
        path_to_txt = filepaths['path_to_txt']
        path_to_evo_folders = filepaths['path_to_evo_folders']
        file = open(path_to_txt, 'a+')  # open file in append mode

    # ================ Add optimal cn_s to the history .txt file ================
    classical_period = find_classical_period(c_ns)['T_p_classical']

    add_header(file, path_to_txt, order_of_potential)      # add header if the file is empty
    add_results(file, Job_int, c_ns, fig_of_merit, constraint_alpha, constraint_time, order_of_potential, classical_period, coeffs) # add the results of the optimization run to the file
    file.close()

    # ================ Add evolution results to .npy files in evolution.out ================
    count = Job_int
    # list of keys you want to save
    keys_to_save = ['tlist', 'T_classical_period', 'xc', 'pc', 'purity2', 'CL', 'alpha', 'peak_CL', 'fig_of_merit', 'c_ns', 'gamma_t','beta', 'cubicity', 'angle_phi', 'blurring', 'condition']
    keys_to_save_loop = ['purity2_loop', 'CL_loop', 'peak_CL_loop', 'alpha_loop']

    if numba_evolution == True:
        name = 'optimal_numba'
        save_npy_data(evo_results_optim, keys_to_save, path_to_evo_folders, name, count)

        alpha_loop = evo_results_optim['alpha_loop']
        if len(alpha_loop) > 0:  # save loop results if available
            save_npy_data(evo_results_optim, keys_to_save_loop, path_to_evo_folders, name, count)

        # save initial results
        name = 'initial_numba'
        save_npy_data(evo_results_init, keys_to_save, path_to_evo_folders, name, count)

    else:
        # save optimal results
        name = 'optimal'
        save_npy_data(evo_results_optim, keys_to_save, path_to_evo_folders, name, count)

        alpha_loop = evo_results_optim['alpha_loop']
        if len(alpha_loop) > 0:  # save loop results if available
            save_npy_data(evo_results_optim, keys_to_save_loop, path_to_evo_folders, name, count)

        # save initial results
        name = 'initial'
        save_npy_data(evo_results_init, keys_to_save, path_to_evo_folders, name, count)

    alpha_loop = evo_results_init['alpha_loop']
    if len(alpha_loop) > 0:  # save loop results if available
        save_npy_data(evo_results_init, keys_to_save_loop, path_to_evo_folders, name, count)


def save_to_history_dw_comparison(S_Omega, theta, evo_results_dw_comparison, order_of_potential):
    """Save evolution data from the double well with d0=d/10 to the folder history/evolution.out"""

    position_dependent_noise = define_physical_params()['position_dependent_noise']

    filepaths = define_filepaths(S_Omega, theta,order_of_potential)
    path_to_evo_folders = filepaths['path_to_evo_folders']

    create_subfolders(S_Omega, theta, order_of_potential)  # create all required folders if they
    # don't already exist

    # ================ Add evolution results to .npy files in evolution.out ================

    # list of keys you want to save
    keys_to_save = ['tlist', 'T_classical_period', 'xc', 'pc', 'purity2', 'CL', 'alpha',  'peak_CL', 'fig_of_merit', 'c_ns',
                    'gamma_t', 'beta', 'cubicity', 'blurring']

    # save evolution results
    int = 1
    name = 'dw_comparison'
    save_npy_data(evo_results_dw_comparison, keys_to_save, path_to_evo_folders, name, int)

def import_evolution_from_history(S_Omega, theta, name, count, Job_int, order_of_potential, double_well_comparison=False, optimization_finished=False):
    phys_params = define_physical_params()
    filepaths = define_filepaths(S_Omega, theta, order_of_potential)
    if optimization_finished==True:
        path_to_evo_folders = filepaths['path_to_evo_folders_final']
        path_to_txt = filepaths['path_to_txt_final']
    else:
        path_to_evo_folders = filepaths['path_to_evo_folders']
        path_to_txt = filepaths['path_to_txt']

    if not os.path.exists(path_to_evo_folders):  # check that the folder exists and exit the function if it doesn't
        message = "Optimization results not available (folder '" + path_to_evo_folders + "' not found)."
        print(Fore.RED + message + Fore.BLACK)
        return

    # list of keys you want to load
    keys_to_load = ['tlist', 'T_classical_period', 'xc', 'pc', 'purity2', 'CL',  'alpha', 'peak_CL',  'fig_of_merit', 'c_ns', 'gamma_t', 'beta', 'cubicity','angle_phi', 'blurring', 'condition']

    results = load_npy_data(keys_to_load, path_to_evo_folders, name, Job_int)  # load results (dictionary)

    # load loop results if available, return empty dict otherwise
    results_loop = {'purity2_loop': {}, 'CL_loop': {}, 'peak_CL_loop': {}, 'alpha_loop': {}}

    results.update(results_loop)                    # add loop results to the output dictionary
    results.update({'S_Omega': S_Omega})    # add Gamma_Omega value to the output dictionary

    if order_of_potential=='four' and double_well_comparison==False:

        all_list = np.loadtxt(path_to_txt, delimiter='\t')

        if count > 1:  #if the number of lines in the .txt file is larger than 1, import the results
            job_list = all_list[:, 0]
            line = np.where(job_list == Job_int)[0]
            d0 = all_list[line, 1]
            results.update({'d0': d0})  # add d0 value to the output dictionary
            a = all_list[line, 2]
            results.update({'a': a})  # add g value to the output dictionary
            b = all_list[line, 3]
            results.update({'b': b})  # add g value to the output dictionary
            c = all_list[line, 4]
            results.update({'c': c})  # add g value to the output dictionary

        else: #if the number of lines in the .txt file is 1, import the results
            d0 = all_list[1]
            results.update({'d0': d0})  # add d0 value to the output dictionary
            a = all_list[2]
            results.update({'a': a})  # add g value to the output dictionary
            b = all_list[3]
            results.update({'b': b})  # add g value to the output dictionary
            c = all_list[4]
            results.update({'c': c})  # add g value to the output dictionary
    else:
        results.update({'d0': {} })
        results.update({'g' : {} })
        results.update({'a' : {} })
        results.update({'b' : {} })
        results.update({'c' : {} })
    return results

def find_best_run(file_path, count):
    """Return the index of the best optimization run stored in the history text file at file_path."""

    if not os.path.exists(file_path):  # check that the file exist, print a message and exit the function if not
        message = """Optimization results not available ('""" + file_path + """' not found)."""
        print(Fore.RED + message + Fore. BLACK)
        return

    all_list = np.loadtxt(file_path, delimiter='\t')
    if count > 1:
        job_list = all_list[:, 0]
        xi_list = np.array(all_list.T[-1])
        #Find maximum fig of merit and return line
        line_idx_best = np.argmax(xi_list) + 1
        #Find value of iteration run corresponding to the maximum
        idx_best = int(job_list[line_idx_best-1])
        print("Job index of the optimal results:")
        print(idx_best)
    else:
        idx_best = int(all_list[0])
        print("Job index of the optimal results:")
        print(idx_best)
    return idx_best

def save_best_run():
    physical_params = define_physical_params()
    S_Omega_exponent_list = physical_params['S_Omega_exp_list']
    order_of_potential = physical_params['order_of_potential']
    theta_list = physical_params['theta_list']

    for theta in theta_list:
        for i, S_Omega_exponent in enumerate(S_Omega_exponent_list):
            S_Omega = 10 ** (S_Omega_exponent)

            #File path to the history folder
            path_to_txt = define_filepaths(S_Omega, theta, order_of_potential)['path_to_txt']

            if not os.path.exists(path_to_txt):
                print(
                    Fore.RED + f"S_Omega_exponent = {S_Omega_exponent}: Final optimization results not available (file '" + path_to_txt + "' does not exist)." + Fore.BLACK)
                continue  # skip the rest of the loop if  results do not exist

            count = count_lines(path_to_txt)  # count how many lines in the history file

            idx_best = find_best_run(path_to_txt, count)
            print(f"Index (Job_int) of the best optimization run: {idx_best}")
            results = import_evolution_from_history(S_Omega, theta, "optimal", count, idx_best, order_of_potential)
            results_initial = import_evolution_from_history(S_Omega, theta, "initial", count, idx_best, order_of_potential)
            optimized_c_ns = results['c_ns']
            optimized_fig_of_merit = results['fig_of_merit']
            optimized_d0 = results['d0'][0]
            optimized_a  = results['a'][0]
            optimized_b  = results['b'][0]
            optimized_c  = results['c'][0]

            optimized_coeffs   = np.array([optimized_d0, optimized_a, optimized_b, optimized_c])

            save_to_history(idx_best, S_Omega, theta, optimized_c_ns, optimized_fig_of_merit, 1, 1, results, results_initial, order_of_potential='four',
                            coeffs=optimized_coeffs, optimization_finished=True )

def count_lines(file_path):
    """Count the lines in the history text file at file_path."""

    with open(file_path, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    return count

def calculateNegativity():

    physical_params = define_physical_params()
    N = physical_params['N']
    d = physical_params['d']
    d_i = physical_params['d_i']
    chi = physical_params['chi']
    d_exponent = int(np.log10(d))
    order_of_potential = physical_params['order_of_potential']
    theta_list = physical_params['theta_list']
    potential_types = physical_params['potential_types']

    opt_params_def = define_optimization_params_quartic('DW')
    cost_func_type = opt_params_def["cost_func_type"]

    S_Omega_exponent_list = physical_params['S_Omega_exp_list']
    for theta in theta_list:
        for i, S_Omega_exponent in enumerate(S_Omega_exponent_list):
            S_Omega = 10 ** (S_Omega_exponent)

            path_to_txt = define_filepaths(S_Omega, theta, order_of_potential)['path_to_txt_final']

            if not os.path.exists(path_to_txt):
                print(
                    Fore.RED + f"S_Omega_exponent = {S_Omega_exponent}: Final optimization results not available (file '" + path_to_txt + "' does not exist)." + Fore.BLACK)


            count = count_lines(path_to_txt)  # count how many lines in the history file
            idx_best = find_best_run(path_to_txt, count)  # returns None if double well results not available
            print(f"Index (Job_int) of the best optimization run: {idx_best}")

            results = import_evolution_from_history(S_Omega, theta, "optimal", count, idx_best, order_of_potential,
                                                    optimization_finished=True)

            # define all relevant lists:
            tlist = results['tlist']  # list of times omega*t
            T_classical_period = results['T_classical_period']  # classical period
            print(f"Classical period for S/Omega = {S_Omega} is  T_c*omega = {T_classical_period} ")
            tlist_norm = tlist / T_classical_period  # normalized time list
            purity = results['purity2']
            cubicity = results['cubicity']
            coh_len = results['CL']
            gamma_t = results['gamma_t']
            fig_of_merit = results['fig_of_merit']
            gamma_t_Omega = gamma_t * chi

            if cost_func_type == 'cubicity_purity':
                fig_of_merit_all_t = np.abs(purity * cubicity)
            elif cost_func_type == 'coherence_length':
                fig_of_merit_all_t = coh_len
            else:
                print("Error")

            index_max  = np.argmax(fig_of_merit_all_t)

            purity_max = purity[index_max]
            k = cubicity[index_max]
            print(f"Index_max:{index_max} out of {len(fig_of_merit_all_t)}")

            print(f"Purity= {purity_max}, Cubicity = {k}, Figure of merit= {fig_of_merit}" )

            def airy(x,p):
                return scipy.special.airy(
                1 / (6 * k) ** (1 / 3) * (-(x - 6 * k * p ** 2) + 1 / (4 *purity_max**2 * 6 * k))) [0]

            wigner = lambda x,p: np.sqrt(2 * purity_max/ (np.pi * 4)) * (1 / (6 * k)) ** (1 / 3) * np.exp(
                -p ** 2 *purity_max/ 2 + 1 / (12 *purity_max**3 * (6 * k) ** 2) - 1 / (2 *purity_max* 6 * k) * (x - 6 * k * p ** 2))  * airy(x,p)

            cm = 1 / 2.54
            sz = (15 * cm, 10 * cm)


