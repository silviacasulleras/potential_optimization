import os
from scipy.special import factorial
import scipy
from colorama import Fore
from pylab import *
from modules.output import  import_evolution_from_history, find_best_run, count_lines, define_filepaths
from modules.parameters import define_plot_params, define_physical_params, define_optimization_params_quartic


def plot_fig_of_merit_vs_S(logscale=False, optimization_finished=False):
    """Plot optimizations results for different S values on the same plot. The functions check that the results for a
    given S value exist. If not, it prints a warning message and skips to the next value of S."""

    physical_params = define_physical_params()
    N = physical_params['N']
    d = physical_params['d']
    d_i = physical_params['d_i']
    chi = physical_params['chi']
    d_exponent = int(np.log10(d))
    order_of_potential = physical_params['order_of_potential']
    theta_list = physical_params['theta_list']

    S_Omega_exponent_list = physical_params['S_Omega_exp_list']
    position_dependent_noise = physical_params['position_dependent_noise']


    opt_params_def = define_optimization_params_quartic('DW')
    lb = opt_params_def["lb"]
    ub = opt_params_def["ub"]
    cost_func_type = opt_params_def["cost_func_type"]

    Npoints = len(S_Omega_exponent_list)

    plot_params = define_plot_params()
    sz = plot_params['sz']
    clr_opt_1 = plot_params['clr_opt_1']
    clr_init = plot_params['clr_init']
    colors = plt.cm.viridis(np.linspace(0, 1, Npoints))  # define a color scheme
    save_as = plot_params['save_as']
    use_tex = plot_params['use_tex']
    font = plot_params['font']

    # ======================== generate figure template ========================

    cm = 1 / 2.54
    sz = (15* cm, 16* cm)
    matplotlib.rcParams.update({'font.size': 12})

    fig = plt.figure(figsize=(sz[0], sz[1]),dpi=300)

    #plt.subplots_adjust(top=0.93, bottom=0.095, left=0.04, right=0.955, hspace=0.255, wspace=0.2)
    # plt.subplots_adjust(top=0.97, bottom=0.05, left=0.02, right=0.975, hspace=0.1, wspace=0.1)

    ax5 = fig.add_subplot(3, 2, 2)
    ax5.set_xlabel("$ S_j\\Omega $",labelpad=0)
    cost_func_type = opt_params_def["cost_func_type"]
    if cost_func_type == 'cubicity_purity':
        ax5.set_ylabel("$\max K(t) $",labelpad=0)
    elif cost_func_type == 'coherence_length_purity_full':
        ax5.set_ylabel("$\max \\xi(t) \\mathcal{P}(T_c) $",labelpad=0)
    elif cost_func_type == 'coherence_length':
        ax5.set_ylabel("$\max_t \\xi(t) $",labelpad=0)
    ax5.set_xscale("log")
    if logscale == True:
        ax5.set_yscale("log")
        #ax5.set_ylim(0.5*1e2,1e5)
    ax5.tick_params(direction='in', which='minor')
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid()

    if cost_func_type=='coherence_length':
        left, bottom, width, height = [0.82, 0.882, 0.15, 0.1]
        ax6 = fig.add_axes([left, bottom, width, height])
        ax6.set_xlabel("$ S_j\\Omega $",fontsize=10,labelpad=0)
        ax6.set_ylabel("$ d_0/d $",fontsize=10,labelpad=-2)
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        ax6.set_xscale("log")
        ax6.tick_params(direction='in')
        plt.grid()

    ax1 = fig.add_subplot(3, 2, 3)
    ax1.set_xlabel("$ t/T_c $")
    ax1.set_ylabel("$\\Gamma(t)/\\Omega$", labelpad = 2)
    ax1.set_yscale("log")
    if cost_func_type == 'cubicity_purity':
        ax1.set_ylim(1e-14, 1e-4)
    else:
        ax1.set_ylim(1e-16, 1e-4)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    ax1.tick_params(direction='in')
    plt.grid()

    ax2 = fig.add_subplot(3, 2, 4)
    ax2.set_xlabel("$ t/T_c $")
    ax2.set_ylabel("$\\Gamma(t)/\\Omega$", labelpad=2)
    ax2.set_yscale("log")
    if cost_func_type == 'cubicity_purity':
        ax2.set_ylim(1e-14, 1e-4)
    else:
        ax2.set_ylim(1e-16, 1e-4)
    ax2.tick_params(direction='in')
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid()

    ax3 = fig.add_subplot(3, 2, 5)
    ax3.set_xlabel("$ t/T_c $")
    ax3.set_yscale("log")
    ax3.tick_params(direction='in')
    if cost_func_type == 'cubicity_purity':
        ax3.set_ylim(1e-5, 1.5*1e4)
        ax3.set_ylabel("$K(t) $")
    elif cost_func_type == 'coherence_length':
        ax3.set_ylabel("$\\xi(t)/x_0 $")
        ax3.set_ylim(2, 1e5)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid()


    ax4 = fig.add_subplot(3, 2, 6)
    ax4.set_xlabel("$ t/T_c $")
    ax4.set_yscale("log")
    ax4.tick_params(direction='in')
    if cost_func_type == 'cubicity_purity':
        ax4.set_ylim(1e-5, 1e4)
        ax4.set_ylabel("$K(t)$")
    elif cost_func_type == 'coherence_length':
        ax4.set_ylabel("$\\xi(t)/x_0 $")
        ax4.set_ylim(2, 1e5)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid()

    fig_of_merit_list_theta_0 = []
    fig_of_merit_list_theta_pi2 = []
    S_Omega_list_theta_0 = []
    S_Omega_list_theta_pi2 =[]

    d0_list_theta_0 = []
    d0_list_theta_pi2 = []

    # ======================== import results for all Ss and add them to plots ========================
    for theta in theta_list:
        for i, S_Omega_exponent in enumerate(S_Omega_exponent_list):
            S_Omega = 10 ** (S_Omega_exponent)

            #File path to the history folder
            if optimization_finished==True:
                path_to_txt = define_filepaths(S_Omega, theta, order_of_potential)['path_to_txt_final']
            else:
                path_to_txt = define_filepaths(S_Omega, theta, order_of_potential)['path_to_txt']
            if not os.path.exists(path_to_txt):
                print(
                    Fore.RED + f"S_Omega_exponent = {S_Omega_exponent}: Final optimization results not available (file '" + path_to_txt + "' does not exist)." + Fore.BLACK)
                continue  # skip the rest of the loop if  results do not exist

            count = count_lines(path_to_txt)  # count how many lines in the history file
            idx_best   = find_best_run(path_to_txt, count)  # returns None if double well results not available
            print(f"Index (Job_int) of the best optimization run: {idx_best}")

            results = import_evolution_from_history(S_Omega, theta, "optimal", count, idx_best, order_of_potential, optimization_finished=optimization_finished)

            # define all relevant lists:
            tlist = results['tlist']  # list of times omega*t
            T_classical_period = results['T_classical_period']  # classical period
            print(f"Classical period for S/Omega = {S_Omega} is  T_c*omega = {T_classical_period} ")
            tlist_norm = tlist / T_classical_period  # normalized time list
            purity = results['purity2']
            cubicity = results['cubicity']
            coh_len = results['CL']
            gamma_t = results['gamma_t']
            gamma_t_Omega = gamma_t*chi
            d0 = results['d0']

            if cost_func_type=='cubicity_purity':
                fig_of_merit_all_t = np.abs(purity * cubicity)
            elif cost_func_type == 'coherence_length':
                fig_of_merit_all_t = coh_len
            else:
                print("Error")

            fig_of_merit = results['fig_of_merit']

            if theta==0:
                S_Omega_list_theta_0.append(S_Omega)
                fig_of_merit_list_theta_0.append(fig_of_merit)
                d0_list_theta_0.append(d0/d)
            else:
                S_Omega_list_theta_pi2.append(S_Omega)
                fig_of_merit_list_theta_pi2.append(fig_of_merit)
                d0_list_theta_pi2.append(d0/d)

            S_Omega_exp = int(np.log10(S_Omega))


            if theta==0:
                ax1.plot(tlist_norm, gamma_t_Omega, color=colors[Npoints - 1 - i], linestyle='-', label = f"$S_j\\Omega =10^{{{S_Omega_exp}}} $")
                ax3.plot(tlist_norm, fig_of_merit_all_t, color=colors[Npoints - 1 - i], linestyle='-', label = f'$S_j\\Omega =10^{{{S_Omega_exp}}}$')
            else:
                ax2.plot(tlist_norm, gamma_t_Omega, color=colors[Npoints - 1 - i], linestyle='-', label = f'$S_j\\Omega =10^{{{S_Omega_exp}}}$')
                ax4.plot(tlist_norm, fig_of_merit_all_t, color=colors[Npoints - 1 - i], linestyle='-', label = f'$S_j\\Omega =10^{{{S_Omega_exp}}}$')


#
    # ======================== generate scatter plot xi vs. S ========================

    ax5.plot(S_Omega_list_theta_0, fig_of_merit_list_theta_0, color='blue', linestyle='-', label='$j=1$')
    ax5.plot(S_Omega_list_theta_pi2, fig_of_merit_list_theta_pi2, color='orange',linestyle='--', label='$j=2$')

    if cost_func_type=='coherence_length':
        ax6.plot(S_Omega_list_theta_0, d0_list_theta_0, color='blue', linestyle='-')
        ax6.plot(S_Omega_list_theta_pi2, d0_list_theta_pi2, color='orange', linestyle='--')
        ax6.scatter(S_Omega_list_theta_pi2, d0_list_theta_pi2, color='orange', marker='o', s=15)
        ax6.scatter(S_Omega_list_theta_0, d0_list_theta_0, color='blue', marker='o', s=15)

    ax5.scatter(S_Omega_list_theta_pi2, fig_of_merit_list_theta_pi2, color='orange', marker='o', s=20)
    ax5.scatter(S_Omega_list_theta_0, fig_of_merit_list_theta_0, color='blue', marker='o', s=20)

    plt.grid()

    ax5.legend(prop={'size': 8},loc='lower left')
    ax3.grid()
    ax4.grid()

    ax3.grid()
    fig.tight_layout(pad=0.5)

    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, ncol=3,prop={'size': 9}, bbox_to_anchor=(0.5, 0), loc='lower center')

    plt.subplots_adjust(bottom=0.17)

    plt.savefig(f'plots/plot_article_six_plots.' + save_as,
                    bbox_inches='tight')
    plt.show()


