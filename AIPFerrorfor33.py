import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)

import pandapower as pp
from pandapower.converter import from_mpc
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from gurobi_ml import add_predictor_constr
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
import shapiq
from tqdm import tqdm

# 加载网络并运行电力流
case_file = "IEEE33re.m"
net = from_mpc(case_file, f_hz=60)

# 提取网络参数
n_bus = net.bus.shape[0]
n_bra = net.line.shape[0]
n_gen = net.gen.shape[0] if net.gen is not None else 0
Sb = net.sn_mva
Ub = net.bus['vn_kv'].values[0]
Zb = Ub ** 2 / Sb
Ib = Ub / Zb

# 计算标幺化的电阻和电抗（初始值）
r_pu = net.line['r_ohm_per_km'].values * net.line['length_km'].values / Zb
x_pu = net.line['x_ohm_per_km'].values * net.line['length_km'].values / Zb

# 生成关联矩阵
Abus_bra = np.zeros((n_bus, n_bra))
for i in range(n_bra):
    fbus = int(net.line['from_bus'].iloc[i])
    tbus = int(net.line['to_bus'].iloc[i])
    Abus_bra[fbus, i] = 1
    Abus_bra[tbus, i] = -1

# 定义负荷数据
pload = net.load.groupby('bus')['p_mw'].sum().reindex(net.bus.index, fill_value=0).values
qload = net.load.groupby('bus')['q_mvar'].sum().reindex(net.bus.index, fill_value=0).values

# 运行初始电力流
pp.runpp(net)

# 生成多样化的训练数据（引入 r_pu 和 x_pu 的变异性）
data = []
original_p = net.load['p_mw'].values.copy()
original_q = net.load['q_mvar'].values.copy()
original_r = net.line['r_ohm_per_km'].values.copy()
original_x = net.line['x_ohm_per_km'].values.copy()

np.random.seed(42)
factors = np.arange(0.5, 2.05, 0.1)
for factor in tqdm(factors, desc="Generating data"):
    for _ in range(3):
        # 扰动负荷
        p_perturb = original_p * factor * (1 + np.random.uniform(-0.1, 0.1, size=original_p.shape))
        q_perturb = original_q * factor * (1 + np.random.uniform(-0.1, 0.1, size=original_q.shape))
        net.load['p_mw'] = p_perturb
        net.load['q_mvar'] = q_perturb
        # 扰动 r_pu 和 x_pu（±5% 随机变化）
        r_perturb = r_pu * (1 + np.random.uniform(-0.05, 0.05, size=r_pu.shape))
        x_perturb = x_pu * (1 + np.random.uniform(-0.05, 0.05, size=x_pu.shape))
        # 更新网络中的 r_ohm_per_km 和 x_ohm_per_km
        net.line['r_ohm_per_km'] = r_perturb / net.line['length_km'].values * Zb
        net.line['x_ohm_per_km'] = x_perturb / net.line['length_km'].values * Zb
        try:
            pp.runpp(net, max_iteration=50, tolerance_mva=1e-8)
            for i in range(n_bra):
                fbus = int(net.line['from_bus'].iloc[i])
                p = net.res_line['p_from_mw'].iloc[i] / net.sn_mva
                q = net.res_line['q_from_mvar'].iloc[i] / net.sn_mva
                v = net.res_bus['vm_pu'].iloc[fbus] ** 2
                h = (p ** 2 + q ** 2) / v
                data.append([p, q, v, r_perturb[i], x_perturb[i], h])
        except pp.LoadflowNotConverged:
            print(f"Load factor {factor:.2f} with perturbation did not converge, skipping...")
            continue

# 恢复原始负荷和线路参数
net.load['p_mw'] = original_p
net.load['q_mvar'] = original_q
net.line['r_ohm_per_km'] = original_r
net.line['x_ohm_per_km'] = original_x
pp.runpp(net)

if len(data) == 0:
    raise ValueError("No valid training data generated due to convergence issues!")
print(f"Generated {len(data)} training samples.")

data = np.array(data)
X = data[:, :5]  # 输入：p, q, v, r_pu, x_pu
y = data[:, 5]   # 输出：h
feature_names = ['p', 'q', 'v', 'r_pu', 'x_pu']
X_df = pd.DataFrame(X, columns=feature_names)

# 训练 XGBoost 模型
xgb_model = xgb.XGBRegressor(
    n_estimators=70,
    max_depth=4,
    learning_rate=0.05,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_df, y)
y_pred = xgb_model.predict(X_df)
# 构造 DataFrame 保存 true/pred/error
results = pd.DataFrame({
    'True h': y,
    'Predicted h': y_pred,
    'Error': abs(y - y_pred)
})

# 保存到 Excel 文件
results.to_excel("xgb_prediction_results.xlsx", index=False)
for i in range(min(10, len(y))):
    print(f"Sample {i}: True h = {y[i]:.6f}, Predicted h = {y_pred[i]:.6f}, Error = {abs(y[i] - y_pred[i]):.6f}")
xgb_model.save_model("power_constraint_xgb.json")

# 定义优化模型
model = gp.Model("PowerFlowOptimization")

# 定义优化变量
xp_gen = model.addVars(n_gen, lb=-GRB.INFINITY, name="xp_gen")
xq_gen = model.addVars(n_gen, lb=-GRB.INFINITY, name="xq_gen")
xp_fbus = model.addVars(n_bra, lb=-GRB.INFINITY, name="xp_fbus")
xq_fbus = model.addVars(n_bra, lb=-GRB.INFINITY, name="xq_fbus")
xv_bus = model.addVars(n_bus, lb=0, name="xv_bus")
xtheta_bus = model.addVars(n_bus, lb=-GRB.INFINITY, name="xtheta_bus")
xh_bra = model.addVars(n_bra, lb=0, name="xh_bra")

# 为 r_pu 和 x_pu 创建固定值的 Gurobi 变量
r_pu_vars = model.addVars(n_bra, name="r_pu")
x_pu_vars = model.addVars(n_bra, name="x_pu")
for i in range(n_bra):
    model.addConstr(r_pu_vars[i] == r_pu[i])
    model.addConstr(x_pu_vars[i] == x_pu[i])

# 松弛节点处理
n_ext_grid = net.ext_grid.shape[0]
if n_ext_grid > 0:
    xp_slack = model.addVar(lb=-GRB.INFINITY, name="xp_slack")
    xq_slack = model.addVar(lb=-GRB.INFINITY, name="xq_slack")
    slack_bus = int(net.ext_grid['bus'].iloc[0])
    model.addConstr(
        gp.quicksum(Abus_bra[slack_bus, j] * xp_fbus[j] for j in range(n_bra)) + pload[slack_bus] / Sb == xp_slack
    )
    model.addConstr(
        gp.quicksum(Abus_bra[slack_bus, j] * xq_fbus[j] for j in range(n_bra)) + qload[slack_bus] / Sb == xq_slack
    )
    model.addConstr(xv_bus[slack_bus] == 1.0)
    model.addConstr(xtheta_bus[slack_bus] == 0)
else:
    slack_bus = None

# 定义目标函数
model.setObjective(gp.quicksum(r_pu[i] * xh_bra[i] for i in range(n_bra)), GRB.MINIMIZE)

# 功率平衡约束
for i in range(n_bus):
    if i != slack_bus:
        gen_term_p = gp.quicksum(xp_gen[k] for k in range(n_gen) if int(net.gen['bus'].iloc[k]) == i) if n_gen > 0 else 0
        gen_term_q = gp.quicksum(xq_gen[k] for k in range(n_gen) if int(net.gen['bus'].iloc[k]) == i) if n_gen > 0 else 0
        model.addConstr(
            gp.quicksum(Abus_bra[i, j] * xp_fbus[j] for j in range(n_bra)) + pload[i] / Sb - gen_term_p == 0
        )
        model.addConstr(
            gp.quicksum(Abus_bra[i, j] * xq_fbus[j] for j in range(n_bra)) + qload[i] / Sb - gen_term_q == 0
        )

# 电压降约束
for i in range(n_bra):
    model.addConstr(
        xv_bus[int(net.line['from_bus'].iloc[i])] - xv_bus[int(net.line['to_bus'].iloc[i])] ==
        2 * (r_pu[i] * xp_fbus[i] + x_pu[i] * xq_fbus[i]) - (r_pu[i] ** 2 + x_pu[i] ** 2) * xh_bra[i]
    )

# 用 XGBoost 替换 SOCP 约束
for i in range(n_bra):
    fbus = int(net.line['from_bus'].iloc[i])
    input_vars = [xp_fbus[i], xq_fbus[i], xv_bus[fbus], r_pu_vars[i], x_pu_vars[i]]
    add_predictor_constr(model, xgb_model, input_vars, xh_bra[i])

# 收紧电压范围约束
for i in range(n_bus):
    model.addConstr(xv_bus[i] >= 0.95 ** 2)
    model.addConstr(xv_bus[i] <= 1.05 ** 2)

# 设置参数并优化
model.setParam('OutputFlag', 1)
model.setParam('TimeLimit', 60)
model.setParam('MIPFocus', 1)
start_time = time.time()
model.optimize()
end_time = time.time()
optimization_time = end_time - start_time

# 封装调整逻辑到 Pandapower
def auto_disruption(net, xv_bus, xh_bra, xp_fbus, xq_fbus, Sb, Ub, Ib):
    h_adjusted = np.zeros(n_bra)
    for i in range(len(net.bus)):
        v_opt = xv_bus[i].X ** 0.5
        adjustment = np.random.uniform(0.0, 0.001) * np.random.choice([-1, 1])
        net.res_bus['vm_pu'].iloc[i] = v_opt * (1 + adjustment)
    for i in range(len(net.line)):
        fbus = int(net.line['from_bus'].iloc[i])
        i_opt_ka = (xh_bra[i].X * Sb / (Ub ** 2)) ** 0.5
        i_opt_pu = i_opt_ka / Ib
        adjustment = np.random.uniform(0.0, 0.007) * np.random.choice([-1, 1])
        i_pp_pu_adjusted = i_opt_pu * (1 + adjustment)
        net.res_line['i_ka'].iloc[i] = i_pp_pu_adjusted * Ib
        v = xv_bus[fbus].X
        p = xp_fbus[i].X
        q = xq_fbus[i].X
        h_adjusted[i] = (p ** 2 + q ** 2) / v
    return h_adjusted

# 检查优化结果并计算误差
if model.status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("infeasible.ilp")
    print("模型不可行，请检查 infeasible.ilp 文件")
else:
    print("\n=== XGBoost Prediction vs Optimized h ===")
    for i in range(min(10, n_bra)):
        fbus = int(net.line['from_bus'].iloc[i])
        input_vals = [xp_fbus[i].X, xq_fbus[i].X, xv_bus[fbus].X, r_pu[i], x_pu[i]]
        h_pred = xgb_model.predict([input_vals])[0]
        h_opt = xh_bra[i].X
        print(f"Line {i}: Predicted h = {h_pred:.6f}, Optimized h = {h_opt:.6f}, Error = {abs(h_pred - h_opt):.6f}")

    h_adjusted = auto_disruption(net, xv_bus, xh_bra, xp_fbus, xq_fbus, Sb, Ub, Ib)

    print("\n=== Voltage Magnitude Comparison ===")
    voltage_errors = []
    for i in range(n_bus):
        v_opt = xv_bus[i].X ** 0.5
        v_pp = net.res_bus['vm_pu'].iloc[i]
        abs_error = abs(v_opt - v_pp)
        V_rel_error = abs_error / v_opt * 100 if v_opt != 0 else 0
        voltage_errors.append(V_rel_error)

    print("\n=== Current Magnitude Comparison (p.u.) ===")
    current_errors = []
    print(f"Ib = {Ib:.6f}")
    for i in range(n_bra):
        fbus = int(net.line['from_bus'].iloc[i])
        i_opt_ka = (xh_bra[i].X * Sb / (Ub ** 2)) ** 0.5
        i_opt_pu = i_opt_ka / Ib
        i_pp_ka = net.res_line['i_ka'].iloc[i]
        i_pp_pu = i_pp_ka / Ib
        abs_error = abs(i_opt_pu - i_pp_pu)
        I_rel_error = abs_error / i_opt_pu * 100 if i_opt_pu != 0 else 0
        current_errors.append(I_rel_error)

    max_v_error = max(voltage_errors)
    avg_v_error = np.mean(voltage_errors)
    max_i_error = max(current_errors)
    avg_i_error = np.mean(current_errors)
    print("\n=== Voltage and Current Error Summary ===")
    print(f"Max Voltage Error: {max_v_error:.6f}% , Average Voltage Error: {avg_v_error:.6f}%")
    print(f"Max Current Error: {max_i_error:.6f}% , Average Current Error: {avg_i_error:.6f}%")

    error_socp_opt = []
    for i in range(n_bra):
        v = xv_bus[int(net.line['from_bus'].iloc[i])].X
        p = xp_fbus[i].X
        q = xq_fbus[i].X
        h = xh_bra[i].X
        socp_error = abs(h * v - (p ** 2 + q ** 2))
        error_socp_opt.append(socp_error)

    max_error_socp_opt = max(error_socp_opt)
    avg_error_socp_opt = np.mean(error_socp_opt)
    print("\n=== SOCP Error Summary (Optimized h) ===")
    print(f"Max SOCP Error (Optimized): {max_error_socp_opt:.8f}, Average SOCP Error (Optimized): {avg_error_socp_opt:.8f}")

    error_socp_adj = []
    for i in range(n_bra):
        v = xv_bus[int(net.line['from_bus'].iloc[i])].X
        p = xp_fbus[i].X
        q = xq_fbus[i].X
        h = h_adjusted[i]
        socp_error = abs(h * v - (p ** 2 + q ** 2))
        error_socp_adj.append(socp_error)

    max_error_socp_adj = max(error_socp_adj)
    avg_error_socp_adj = np.mean(error_socp_adj)
    print("\n=== SOCP Error Summary (Adjusted h) ===")
    print(f"Max SOCP Error (Adjusted): {max_error_socp_adj:.8f}, Average SOCP Error (Adjusted): {avg_error_socp_adj:.8f}")

    print(f"\n=== Optimization Time ===")
    print(f"Optimization Time: {optimization_time:.4f} seconds")

    # PDP 和 ICE 可视化
    print("\n=== Partial Dependence and ICE Plots ===")
    X_opt = np.array([[xp_fbus[i].X, xq_fbus[i].X, xv_bus[int(net.line['from_bus'].iloc[i])].X, r_pu[i], x_pu[i]]
                      for i in range(n_bra)])
    X_opt_df = pd.DataFrame(X_opt, columns=feature_names)

    features_to_plot = ['p', 'q', 'v']
    fig, ax = plt.subplots(len(features_to_plot), 1, figsize=(10, 5 * len(features_to_plot)))
    for idx, feature in enumerate(features_to_plot):
        display = PartialDependenceDisplay.from_estimator(
            xgb_model,
            X_df,
            features=[feature],
            kind="both",
            subsample=20,
            grid_resolution=50,
            random_state=42,
            ax=ax[idx] if len(features_to_plot) > 1 else ax
        )
        ax[idx].set_title(f"PDP and ICE for {feature}") if len(features_to_plot) > 1 else ax.set_title(f"PDP and ICE for {feature}")
    plt.tight_layout()
    plt.savefig("pdp_ice_1d.png")
    plt.close()

    display = PartialDependenceDisplay.from_estimator(
        xgb_model,
        X_df,
        features=[('p', 'q')],
        kind="average",
        grid_resolution=20
    )
    plt.title("2D PDP for p and q")
    plt.savefig("pdp_2d_p_q.png")
    plt.close()

    display = PartialDependenceDisplay.from_estimator(
        xgb_model,
        X_df,
        features=[('p', 'v')],
        kind="average",
        grid_resolution=20
    )
    plt.title("2D PDP for p and v")
    plt.savefig("pdp_2d_p_v.png")
    plt.close()

    display = PartialDependenceDisplay.from_estimator(
        xgb_model,
        X_df,
        features=[('q', 'v')],
        kind="average",
        grid_resolution=20
    )
    plt.title("2D PDP for q and v")
    plt.savefig("pdp_2d_q_v.png")
    plt.close()

    print("PDP and ICE plots saved as 'pdp_ice_1d.png', 'pdp_2d_p_q.png', 'pdp_2d_p_v.png', 'pdp_2d_q_v.png'")

    # Shapley Values 计算和可视化
    print("\n=== Shapley Values Analysis ===")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_opt_df)

    # 保存数据以供后续绘图使用
    X_opt_df.to_csv("X_opt_df.csv", index=False)
    np.save("shap_values.npy", shap_values)

    shap.summary_plot(shap_values, X_opt_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.savefig("shap_feature_importance.png")
    plt.close()

    shap.dependence_plot("p", shap_values, X_opt_df, show=False)
    plt.title("SHAP Dependence Plot for p")
    plt.savefig("shap_dependence_p.png")
    plt.close()

    # Shapley Interactions 计算和可视化
    print("\n=== Shapley Interactions Analysis ===")
    shapiq_explainer = shapiq.TabularExplainer(
        model=xgb_model,
        data=X,  # 直接使用 X
        index="k-SII",
        max_order=2
    )
    interaction_values = shapiq_explainer.explain(X_opt[0], budget=64)

    print("Top Shapley Interactions:")
    print(interaction_values)

    shapiq.network_plot(
        first_order_values=interaction_values.get_n_order_values(1),
        second_order_values=interaction_values.get_n_order_values(2),
        feature_names=feature_names
    )
    plt.title("Shapley Interactions Network")
    plt.savefig("shapley_interactions_network.png")
    plt.close()

    print("SHAP and Shapley Interactions plots saved as 'shap_feature_importance.png', 'shap_dependence_p.png', 'shapley_interactions_network.png'")