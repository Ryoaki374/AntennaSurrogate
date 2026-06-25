import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import lib_gp as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats.qmc import LatinHypercube, scale
import time
plt.style.use('./graph_preset.mplstyle')


def plot_learning_curve(df_output, objective_col="Objective"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(
        df_output.index,
        df_output["best"],
        marker='o',
        linestyle='-',
        markersize=4,
        color='tab:blue',
        label='learning curve'
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Best Minimum Value Found ({objective_col})', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')

    metric_col = "Metric" if "Metric" in df_output.columns else "Acq"

    ax2 = ax.twinx()
    ax2.plot(
        df_output.index,
        df_output[metric_col],
        marker='x',
        linestyle=':',
        markersize=4,
        color='tab:green',
        label='Calculated metric'
    )
    ax2.set_ylabel(metric_col, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def fit_length_scale_and_visualize(
    df_output,
    param_names,
    initial_length_scale,
    noise_var,
    kernel_type="RBF",
    bounds=(1e-2, 1e2),
    plot=True,
    objective_col="Objective",
):
    # --- データ行列の構築（元コード準拠） ---
    X_sample = df_output[param_names].values
    y_sample = df_output[[objective_col]].values  # 形状 (n,1) を維持

    # --- length_scaleの最適化（元コード準拠） ---
    opt = minimize(
        fun=gp.negative_log_marginal_likelihood,
        x0=[initial_length_scale],
        args=(X_sample, y_sample, noise_var),
        bounds=[bounds],
    )
    length_scale = float(opt.x[0])
    print(f"Final optimized length_scale: {length_scale:.4f}")

    # --- 共分散行列の算出（元コード準拠） ---
    K = gp.kernel(X_sample, X_sample, length_scale)
    Ky = K + noise_var * np.identity(len(X_sample))
    Ky_inv = np.linalg.inv(Ky)

    # --- 可視化（元コード準拠、任意） ---
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Final Covariance Matrix ({kernel_type} Kernel)")
        plt.show()

    return {
        "length_scale": length_scale,
        "K": K,
        "Ky": Ky,
        "Ky_inv": Ky_inv,
        "opt": opt,
    }

def plot_covariance_matrix(K):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, ax=ax)
    plt.show()

def plot_pdp_ice(
    df_output,
    final_Ky_opt_inv,
    final_optimized_length_scale,
    param_names,
    lower_bounds,
    upper_bounds,
    pdp_grid_resolution=50,
    nrows=2,
    ncols=2,
    figsize=(12, 10),
    dpi=80,
    show=True,
    objective_col="Objective",
):
    
    X_sample = df_output[param_names].values
    y_sample = df_output[[objective_col]].values  # 形状 (n,1) を維持
    n_features = X_sample.shape[1]
    n_plots = min(n_features, nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(-1)  # フラット化

    dim_ranges = []
    pdp_lines = []
    ice_all = []

    for i in range(n_plots):
        ax_pdp = axes[i]
        dim_range = np.linspace(lower_bounds[i], upper_bounds[i], pdp_grid_resolution)

        # ICE: 各サンプルについて次元iのみを走査して予測
        ice_predictions = np.zeros((len(X_sample), pdp_grid_resolution))
        X_temp = np.copy(X_sample)

        for k, val in enumerate(dim_range):
            X_temp[:, i] = val
            K_star = gp.kernel(X_sample, X_temp, final_optimized_length_scale)
            mu_post = K_star.T @ final_Ky_opt_inv @ y_sample
            ice_predictions[:, k] = mu_post.ravel()

        # 個別ICE
        for j in range(len(X_sample)):
            ax_pdp.plot(dim_range, ice_predictions[j, :], color='gray', lw=0.5, alpha=0.5)

        # PDP（ICE平均）
        pdp_line = np.mean(ice_predictions, axis=0)
        ax_pdp.plot(dim_range, pdp_line, color='orange', lw=3, label='PDP (Average)')

        ax_pdp.set_xlabel(f'{param_names[i]}')
        ax_pdp.set_ylabel(f'Partial Dependence ({objective_col})')
        ax_pdp.set_title(f'PDP & ICE for {param_names[i]}')
        ax_pdp.grid(True, linestyle='--', alpha=0.6)

        # 凡例（最初のプロットにICEの代理線を追加）
        if i == 0:
            ice_line_proxy = Line2D([0], [0], color='gray', lw=0.5, alpha=0.5, label='ICE')
            handles, labels = ax_pdp.get_legend_handles_labels()
            handles.append(ice_line_proxy)
            ax_pdp.legend(handles=handles)
        else:
            ax_pdp.legend()

        dim_ranges.append(dim_range)
        pdp_lines.append(pdp_line)
        ice_all.append(ice_predictions)

    # 余ったサブプロットは非表示
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Partial Dependence & Individual Conditional Expectation Plots', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": axes.reshape(nrows, ncols),
        "dim_ranges": dim_ranges,
        "pdp_lines": pdp_lines,
        "ice_all": ice_all,
    }

def plot_integrated_ei(
    df_output,
    final_Ky_opt_inv,
    final_optimized_length_scale,
    param_names,
    lower_bounds,
    upper_bounds,
    ei_grid_resolution=50,
    nrows=2,
    ncols=2,
    figsize=(12, 10),
    dpi=80,
    show=True,
    objective_col="Objective",
):
    X_sample = df_output[param_names].values
    y_sample = df_output[[objective_col]].values  # 形状 (n,1) を維持

    n_features = X_sample.shape[1]
    n_plots = min(n_features, nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(-1)

    dim_ranges_all = []
    integrated_ei_all = []

    for i, ax_ei in enumerate(axes[:n_plots]):
        dim_range = np.linspace(lower_bounds[i], upper_bounds[i], ei_grid_resolution)
        integrated_ei_values = np.empty(ei_grid_resolution, dtype=float)

        X_temp = np.copy(X_sample)

        for t, val in enumerate(dim_range):
            X_temp[:, i] = val
            # 各点のEIを計算し平均
            eis_at_val = [
                gp.expected_improvement(
                    x_point, X_sample, y_sample, final_Ky_opt_inv, final_optimized_length_scale
                )
                for x_point in X_temp
            ]
            integrated_ei_values[t] = np.mean(eis_at_val)

        ax_ei.plot(dim_range, integrated_ei_values, color='purple', lw=2)
        ax_ei.set_xlabel(f'{param_names[i]}')
        ax_ei.set_ylabel('Integrated EI')
        ax_ei.set_title(f'Integrated EI for {param_names[i]}')
        ax_ei.grid(True, linestyle='--', alpha=0.6)

        dim_ranges_all.append(dim_range)
        integrated_ei_all.append(integrated_ei_values)

    # 余った軸は非表示
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Integrated Expected Improvement (Marginal Acquisition Function)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": axes.reshape(nrows, ncols),
        "dim_ranges": dim_ranges_all,
        "integrated_ei_values": integrated_ei_all,
    }

def lcb_for_de_objective(X_sample, y_sample, vector, final_Ky_opt_inv, final_optimized_length_scale, KAPPA_DE = 2.0):
    #X_sample = df_output[param_names].values
    #y_sample = df_output[[objective_col]].values  # 形状 (n,1) を維持
    mean, std_dev = gp.get_posterior(vector, X_sample, y_sample, final_Ky_opt_inv, final_optimized_length_scale)
    return mean - KAPPA_DE * std_dev

def differential_evolution_lcb(
    df_output,
    param_names,
    lower_bounds,
    upper_bounds,
    pop_size=100,
    n_generations=200,
    F=0.8,
    CR=0.8,
    kappa=2.0,
    seed=101,
    Ky_inv=None,
    length_scale=None,
    verbose=True,
    objective_col="Objective",
):
    
    rng = np.random.default_rng(seed)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    de_bounds = np.vstack([lower_bounds, upper_bounds]).T
    D = len(lower_bounds)

    X_sample = df_output[param_names].values
    y_sample = df_output[[objective_col]].values  # 形状 (n,1) を維持

    if verbose:
        print(f"Starting DE with {n_generations} generations "
              f"(optimizing LCB with kappa={kappa})...")

    start_time = time.time()

    # 初期集団（LHS→スケーリング）
    sampler = LatinHypercube(d=D, seed=seed)
    population = scale(sampler.random(n=pop_size), de_bounds[:, 0], de_bounds[:, 1])

    for g in range(n_generations):
        for i in range(pop_size):
            target = population[i]
            # 3個体選択
            idxs = np.delete(np.arange(pop_size), i)
            a, b, c = population[rng.choice(idxs, 3, replace=False)]
            # 変異
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, de_bounds[:, 0], de_bounds[:, 1])
            # 交叉（一様）
            cross = rng.random(D) < CR
            if not np.any(cross):
                cross[rng.integers(0, D)] = True
            trial = np.where(cross, mutant, target)
            # 貪欲置換
            target_fit = lcb_for_de_objective(X_sample, y_sample, target, Ky_inv, length_scale, kappa)
            trial_fit = lcb_for_de_objective(X_sample, y_sample, trial, Ky_inv, length_scale, kappa)
            if trial_fit < target_fit:
                population[i] = trial

    # 結果
    fitness_values = np.array([lcb_for_de_objective(X_sample, y_sample, ind, Ky_inv, length_scale, kappa) for ind in population])
    best_index = int(np.argmin(fitness_values))
    best_solution = population[best_index]
    best_fitness = float(fitness_values[best_index])

    post_mu = post_std = best_lcb = None
    if gp is not None and X_sample is not None and y_sample is not None and Ky_inv is not None and length_scale is not None:
        mu, std = gp.get_posterior(best_solution, X_sample, y_sample, Ky_inv, length_scale)
        post_mu = float(mu)
        post_std = float(std)
        best_lcb = post_mu - kappa * post_std

    end_time = time.time()
    if verbose:
        print("\n--- DE Search Finished ---")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print("\n" + "="*20, " GLOBAL MINIMUM FOUND BY DE (LCB) ", "="*20)
        print("The best parameters found by searching the surrogate model are:")
        for name, val in zip(param_names, best_solution):
            print(f"  {name}: {val:.6f}")
        print("-" * 54)
        if post_mu is not None:
            print(f"Predicted {objective_col} at this point (μ): {post_mu:.6f}")
            print(f"Prediction Uncertainty (σ): {post_std:.6f}")
            print(f"Pessimistic Estimate (LCB = μ - κ*σ): {best_lcb:.6f}")
        else:
            print(f"Best LCB (objective value): {best_fitness:.6f}")
        print("="*54)

    return {
        "best_solution": best_solution,
        "best_index": best_index,
        "best_fitness": best_fitness,
        "population": population,
        "fitness_values": fitness_values,
        "exec_time_sec": end_time - start_time,
        "posterior_mean": post_mu,
        "posterior_std": post_std,
        "best_lcb_value": best_lcb,
    }
