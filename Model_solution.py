# SECTION 0 — IMPORTS, FILE PATHS, CONSTANTS
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pyomo.environ import value
import matplotlib.pyplot as plt
import math
import traceback
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib as mpl



use_kkt     = True
USE_NETWORK = True

BEV_XLSX        = r"C:\Users\HP\Desktop\Code\Inputs\BEV_parameters.xlsx"
TIMESERIES_XLSX = r"C:\Users\HP\Desktop\Code\Inputs\Time_Series.xlsx"


mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

#CONNECTION_MODE = "scaled"   # the connection capacity for the fleet increases with the fleet size
CONNECTION_MODE = "fixed"   # connection capacity limited to one fleet node with 1 MVA.



DELTA_T = 1.0 #TIME STEP

S_BASE_MVA = 10.0 # for conversion to pu scale

# NETWORK VOLTAGE LIMITS
V_base = 12.66 #kV
V_MIN = 0.95**2 # lower limit
V_MAX = 1.05**2 #upper limit

# Congesition Weights
W_TH = 1.0      # weight on thermal congestion 
W_V  = 1.0      # weight on voltage congestion
EPS_CONG = 1e-6 

# Voltage normalization band
DV_BAND = max(EPS_CONG, (V_MAX - V_MIN))  # in pu^2

# Fleet power factor
PF_FLEET = 0.95
TANPHI_F = np.tan(np.arccos(PF_FLEET))
# PV power factor
PF_PV = 0.9
TANPHI = np.tan(np.arccos(PF_PV))

SOC_MIN_DEP = 0.8  # Minimum charging need

BIGM = 1e4

# Congestion thresholds (dimensionless)
THETA = [0.0, 0.05, 0.15, 0.35, 0.7, 1.0]
#Maximum tariff step
LAMBDA_MAX = 1400.0 # in €/pu = 140 €/MWh

# Penalties
S_SOC_MAX = 0.2   # bound for SOC-dynamics slack variable
C_SHORT = 500     # €/MWh- unmet SOC 
C_ENS   = 100000      # €/MWh - Penalty for energy not served
C_CONG  = 5000      # congestion penalty weight (dimensionless)

SOC_SLACK_PEN = 20000 #
C_VOLT  = 4000   # strong penalty for voltage violation
C_THERM = 6000   # strong penalty for thermal overload
C_PVC  = 300    # small PV penalty
#C_REG = 3   # regulation of th DSO behaiour 


#arrays for storing outputs
summary_rows = []
hourly_rows = []
tariff_rows = []



# SECTION 1 - Inputs

bev = pd.read_excel(BEV_XLSX)
bev = bev.dropna(how="all")
bev = bev.dropna(subset=["charger_id"])
bev = bev.dropna(subset=["t_arr", "t_dep"])
bev["charger_id"] = bev["charger_id"].astype(str)

if "bus" not in bev.columns:
    raise ValueError("BEV Excel must contain a 'bus' column.")

bev = bev.dropna(subset=["bus"])
bev["bus"] = bev["bus"].astype(int)

T = list(range(24))

def parse_hour(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    x = str(x).strip()
    return int(x.split(":")[0])

def availability(ta, td, t):
    if ta < td:
        return 1 if (ta <= t < td) else 0
    else:
        return 1 if (t >= ta or t < td) else 0

t_arr_map, t_dep_map = {}, {}

for _, row in bev.iterrows():
    b = str(row["charger_id"])
    t_arr_map[b] = int(row["t_arr"]) % 24
    t_dep_map[b] = int(row["t_dep"]) % 24

B = list(bev["charger_id"].unique())



# Truck → depot bus mapping
bus_of_base = {}
for _, row in bev.iterrows():
    b = str(row["charger_id"])
    bus_of_base[b] = int(row["bus"])
def bev_scalar(b, col):
    if isinstance(b, (list, tuple, np.ndarray)):
        if len(b) != 1:
            raise ValueError(f"bev_scalar got list of length {len(b)}: {b}")
        b = b[0]

    b = str(b)

    if "__rep" in b:
        b = b.split("__rep")[0]

    row = bev.loc[bev["charger_id"] == b]
    if row.empty:
        raise ValueError(f"Vehicle {b} not found in BEV table (charger_id).")

    return float(row[col].iloc[0])


def bev_bus(b):
    if "__rep" in b:
        b = b.split("__rep")[0]
    b = str(b)

    if b not in bus_of_base:
        raise ValueError(f"Vehicle {b} not found in bus mapping.")

    return bus_of_base[b]


for b in B:
    if b not in t_arr_map or b not in t_dep_map:
        raise ValueError(f"Missing t_arr/t_dep for vehicle {b}")

    if not (0 <= t_arr_map[b] <= 23) or not (0 <= t_dep_map[b] <= 23):
        raise ValueError(f"Invalid time for {b}: t_arr={t_arr_map[b]}, t_dep={t_dep_map[b]}")
   
   

# Fleet parameters (all in pu)
Pmax_fleet = sum(bev_scalar(b, "p_max") for b in B)
Emax_fleet = sum(bev_scalar(b, "battery_size") for b in B)
E0_fleet   = sum(bev_scalar(b, "e0") * bev_scalar(b, "battery_size") for b in B)
Etarget_fleet = sum(bev_scalar(b, "e_target") * bev_scalar(b, "battery_size") for b in B)

eta_fleet = float(bev["eta"].mean())

def t_arr(b): return t_arr_map[b]
def t_dep(b): return t_dep_map[b]

a_fleet = {t: int(any(availability(t_arr(b), t_dep(b), t) for b in B)) for t in T}

E_dep_req = {t: 0.0 for t in T}
for b in B:
    dep = (t_dep_map[b] - 1) % 24
    battery = bev_scalar(b, "battery_size")
    E_dep_req[dep] += SOC_MIN_DEP * battery

   
print("Number of vehicles in model:", len(B))
print("Pmax_fleet (pu):", Pmax_fleet)
print("Emax_fleet (pu-energy):", Emax_fleet)


#Aggregation of the fleet 
FLEET_BUS_SINGLE = 18

def build_single_node_aggregates(fleet_scale=1, agg_bus=18):
    fleet_scale = int(fleet_scale or 1)

    # all vehicles aggregated into one fleet
    D = [agg_bus]

    Pmax_D = {agg_bus: fleet_scale * sum(bev_scalar(b, "p_max") for b in B)}
    Emax_D = {agg_bus: fleet_scale * sum(bev_scalar(b, "battery_size") for b in B)}
    E0_D = {agg_bus: fleet_scale * sum(bev_scalar(b, "e0") * bev_scalar(b, "battery_size") for b in B)}
    Etarget_D = {agg_bus: fleet_scale * sum(bev_scalar(b, "e_target") * bev_scalar(b, "battery_size") for b in B)}

    eta_vals = [bev_scalar(b, "eta") for b in B]
    eta_D = {agg_bus: float(np.mean(eta_vals)) if eta_vals else eta_fleet}

    # aggregated hourly availability of the whole fleet
    A_D = {}
    for t in T:
        A_D[(agg_bus, t)] = fleet_scale * sum(
            availability(t_arr_map[b], t_dep_map[b], t) * bev_scalar(b, "p_max")
            for b in B
        )

    # aggregate departure enforcement hour - latest departure among vehicles
    dep_hour = max(t_dep_map[b] for b in B)
    dep_hour_D = {agg_bus: (dep_hour - 1) % 24}

    return D, Pmax_D, Emax_D, E0_D, Etarget_D, eta_D, A_D, dep_hour_D




# Time series data
df_load = pd.read_excel(TIMESERIES_XLSX, sheet_name="load_profile")
df_load.rename(columns={df_load.columns[0]: "time_str"}, inplace=True)
df_load = df_load[df_load["time_str"].notna()].fillna(0.0)
df_load["t"] = df_load["time_str"].apply(parse_hour).astype(int)

bus_cols = [c for c in df_load.columns if c not in ["time_str", "t"]]
df_load_long = df_load.melt(id_vars=["t"], value_vars=bus_cols,
                            var_name="bus", value_name="P_load_pu")
df_load_long["bus"] = df_load_long["bus"].astype(int)


df_Qload = pd.read_excel(TIMESERIES_XLSX, sheet_name="load_Q")
df_Qload.rename(columns={df_Qload.columns[0]: "time_str"}, inplace=True)


df_Qload = df_Qload.dropna(subset=["time_str"])
df_Qload = df_Qload[df_Qload["time_str"].astype(str).str.strip() != ""]
df_Qload = df_Qload.fillna(0.0)
df_Qload["t"] = df_Qload["time_str"].apply(parse_hour).astype(int)

df_pv = pd.read_excel(TIMESERIES_XLSX, sheet_name="PV_profile")

if "bus_id" in df_pv.columns:
    df_pv.rename(columns={"bus_id": "bus"}, inplace=True)

df_pv = df_pv.dropna(subset=["bus", "t"])

df_pv["bus"] = df_pv["bus"].astype(int)
df_pv["t"]   = df_pv["t"].astype(int)


net = pd.read_excel(TIMESERIES_XLSX, sheet_name="network")
net["line_id"]  = net["line_id"].astype(int)
net["from_bus"] = net["from_bus"].astype(int)
net["to_bus"]   = net["to_bus"].astype(int)

slack_bus = int(pd.read_excel(TIMESERIES_XLSX, sheet_name="slack_bus").iloc[0, 0])

df_fleet = pd.read_excel(TIMESERIES_XLSX, sheet_name="fleet_buses")

# PV buses
pv_buses_df = pd.read_excel(TIMESERIES_XLSX, sheet_name="pv_buses")
PV_BUSES = pv_buses_df.iloc[:,0].dropna().astype(int).tolist()

df_fleet["bus"] = pd.to_numeric(df_fleet["bus"], errors="coerce")
df_fleet = df_fleet.dropna(subset=["bus"])
df_fleet["bus"] = df_fleet["bus"].astype(int)

fleet_buses = df_fleet["bus"].tolist()
P_conn_map = {int(r["bus"]): float(r["P_conn"]) for _, r in df_fleet.iterrows()}


I = sorted(set(net["from_bus"]).union(set(net["to_bus"])))
line_ids = list(net["line_id"])

from_bus_map = {int(r["line_id"]): int(r["from_bus"]) for _, r in net.iterrows()}
to_bus_map   = {int(r["line_id"]): int(r["to_bus"])   for _, r in net.iterrows()}
R_map        = {int(r["line_id"]): float(r["R"])      for _, r in net.iterrows()}
X_map        = {int(r["line_id"]): float(r["X"])      for _, r in net.iterrows()}

Smax_map = {int(r["line_id"]): float(r["S_max"]) for _, r in net.iterrows()}


P_load_dict = {(i, t): 0.0 for i in I for t in T}
PV_dict     = {(i, t): 0.0 for i in I for t in T}

for _, r in df_load_long.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        P_load_dict[(i, t)] = float(r["P_load_pu"])


for _, r in df_pv.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        PV_dict[(i,t)] = float(np.nan_to_num(r["PV"], nan=0.0))

Qload_dict = {(i, t): 0.0 for i in I for t in T}
for _, r in df_Qload.iterrows():
    t = int(r["t"])
    if t in T:
        for bus in df_Qload.columns:
            if bus not in ["time_str", "t"]:
                i = int(bus)
                Qload_dict[(i, t)] = float(r[bus])

# price
price_dict = {t: 0.0 for t in T}
try:
    prices_raw = pd.read_excel(TIMESERIES_XLSX, sheet_name="price_profile")
    if "t" not in prices_raw.columns:
        prices_raw.rename(columns={prices_raw.columns[0]: "time_str"}, inplace=True)
        prices_raw["t"] = prices_raw["time_str"].apply(lambda x: int(str(x).split(":")[0]))
    prices_raw["t"] = prices_raw["t"].astype(int)
    prices_raw = prices_raw[prices_raw["t"].isin(T)]
    for _, row in prices_raw.iterrows():
        price_dict[int(row["t"])] = float(row["price"])
except Exception:
    pass

print("Pmax_fleet (pu):", Pmax_fleet)
print("Emax_fleet (pu-energy):", Emax_fleet)

print("Total feeder load peak:",
      max(sum(P_load_dict[i,t] for i in I) for t in T) )

print("Total PV peak:",
      max(sum(PV_dict[i,t] for i in I) for t in T) )


def build_asap_profile(B_exp, base_of, T, eta_fleet, DELTA_T):
    Pb_fix = {(b,t): 0.0 for b in B_exp for t in T}

    for b in B_exp:
        b0 = base_of[b]
        pmax = bev_scalar(b0, "p_max")
        batt = bev_scalar(b0, "battery_size")
        E0   = bev_scalar(b0, "e0") * batt
        Etgt = bev_scalar(b0, "e_target") * batt

        need = max(0.0, Etgt - E0)

        ta = t_arr_map[b0] % 24
        td = t_dep_map[b0] % 24

        avail = []
        if ta < td:
            avail = list(range(ta, td))
        else:
            avail = list(range(ta, 24)) + list(range(0, td))

        for t in avail:
            if need <= 1e-9:
                break
            e_add = eta_fleet * DELTA_T * pmax
            Pb_fix[(b,t)] = pmax
            need -= e_add

    return Pb_fix

def build_uncontrolled_profile(fleet_scale=1):
    P = {t: 0.0 for t in T}

    for rep in range(fleet_scale):
        for b in B:
            ta = t_arr_map[b]
            td = t_dep_map[b]
            pmax = bev_scalar(b, "p_max")
            batt = bev_scalar(b, "battery_size")
            e0 = bev_scalar(b, "e0") * batt
            et = bev_scalar(b, "e_target") * batt
            need = max(0.0, et - e0)

            # energy per hour delivered at pmax:
            e_per_h = eta_fleet * DELTA_T * pmax

            hours = []
            if ta < td:
                hours = list(range(ta, td))
            else:
                hours = list(range(ta, 24)) + list(range(0, td))

            for t in hours:
                if need <= 1e-9:
                    break
                addP = min(pmax, need / max(1e-9, e_per_h) * pmax)  
                addP = pmax
                P[t] += addP
                need -= e_per_h

    return P


print("Peak total load pu:", max(sum(P_load_dict[i,t] for i in I) for t in T))
print("Peak total load MW:", max(sum(P_load_dict[i,t] for i in I) for t in T) * S_BASE_MVA)
print("Peak fleet pu:", Pmax_fleet)
print("Peak fleet MW:", Pmax_fleet * S_BASE_MVA)
print("Price range:", min(price_dict.values()), max(price_dict.values()))

MAX_SIGMA = len(line_ids) * max(Smax_map.values())
MAX_XI    = len(I) * DV_BAND

BIGM_LADDER = W_TH * MAX_SIGMA + W_V * MAX_XI



# SECTION 2 — MODEL BUILDER

def build_model(include_network=True,
                enforce_no_worse=False,
                theta_levels=None,
                bigm_ladder=None,

                use_kkt= True,
                fixed_fleet_profile=None,
                tariff_logic=True,
                MAX_ENS_base=None,
                no_fleet=False,
                fleet_mode="opt",
    
                fixed_Pb_profile=None,
                fleet_scale=None,
                gamma_value=1,
                cong_ref=None,
            ):
   
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.T = pyo.Set(initialize=T, ordered=True)
    t0_global = T[0]
    T_NOT0 = [tt for tt in T if tt != t0_global]
    m.T_NOT0 = pyo.Set(initialize=T_NOT0, ordered=True)
    m.PV_BUSES = pyo.Set(initialize=PV_BUSES)
    if fleet_scale is None:
        fleet_scale = 1
    if cong_ref is None:
        cong_ref = {t: 0.0 for t in T}
    m.CONG_REF = pyo.Param(m.T, initialize=cong_ref, mutable=False)


    # FLEET HANDLING 
    m.base_of = {}
    #.FB = pyo.Set(initialize=fleet_buses)
    if fleet_mode == "none":
        no_fleet = True
        use_kkt = False
        m.D = pyo.Set(initialize=[])
        m.Pfleet = pyo.Param(m.T, initialize={t: 0.0 for t in T}, mutable=False)
    elif fleet_mode == "fixed":
        no_fleet = False
        use_kkt = False
        m.D = pyo.Set(initialize=[])
        if fixed_fleet_profile is None:
            raise ValueError("fleet_mode='fixed' requires fixed_fleet_profile dict {t: Pfleet}")
        m.Pfleet = pyo.Param(m.T, initialize=fixed_fleet_profile, mutable=False)
    elif fleet_mode == "opt":
        no_fleet = False
        D, Pmax_D, Emax_D, E0_D, Etarget_D, eta_D, A_D, dep_hour_D = build_single_node_aggregates(
            fleet_scale=fleet_scale,
            agg_bus=18
        )
        m.Emax_D = Emax_D
        m.D = pyo.Set(initialize=D)
        m.base_of = {}
    else:
        raise ValueError("fleet_mode must be one of: 'none', 'fixed', 'opt'")
   
   

    # PARAMETERS
    m.price = pyo.Param(m.T, initialize=price_dict)

    # UPPER LEVEL — TARIFF LADDER
    m.lambda_tar = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    

    if theta_levels is None:
        theta = THETA          #fallback to global
    else:
        theta = theta_levels   #dynamic thresholds

   
    if not tariff_logic:
        for t in m.T:
            m.lambda_tar[t].fix(0.0)



    if fleet_mode == "opt":
        m.Pd = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
        m.Ed = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
        m.s_short = pyo.Var(m.D, domain=pyo.NonNegativeReals)
        
        m.s_soc = pyo.Var(m.D, m.T, bounds=(-S_SOC_MAX, S_SOC_MAX))
        m.s_soc_abs = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
        t0 = m.T.first()
        
        for d in m.D:
            m.Ed[d, t0].fix(E0_D[d])         # initial SOC fixed -> exclude from KKT stationarity
            m.s_soc[d, t0].fix(0.0)          # no SOC-dynamics slack at initial time
            m.s_soc_abs[d, t0].fix(0.0)

        m.Pfleet = pyo.Expression(m.T, rule=lambda m, t: sum(m.Pd[d, t] for d in m.D))
        
        if CONNECTION_MODE == "fixed":
            P_conn_eff = {18: P_conn_map[18]}
        elif CONNECTION_MODE == "scaled":
            P_conn_eff = {18: fleet_scale * P_conn_map[18]}
        else:
            raise ValueError("Invalid CONNECTION_MODE")
        U_D = {(d, t): min(A_D[(d, t)], P_conn_eff[d]) for d in D for t in T}
        m.U_D = pyo.Param(m.D, m.T, initialize=U_D, mutable=False)
        def p_limit_rule(m, d, t):
            return m.Pd[d, t] <= m.U_D[d, t]
        m.P_LIMIT = pyo.Constraint(m.D, m.T, rule=p_limit_rule)
        
        def soc_dyn_rule(m, d, t):
            if t == m.T.first():
                return pyo.Constraint.Skip
            return m.Ed[d, t] == m.Ed[d, m.T.prev(t)] + eta_D[d] * DELTA_T * m.Pd[d, t] + m.s_soc[d,t]
        m.SOC_DYN = pyo.Constraint(m.D, m.T, rule=soc_dyn_rule)

        m.E_MAX = pyo.Constraint(m.D, m.T_NOT0, rule=lambda m, d, t: m.Ed[d, t] <= Emax_D[d])
        def departure_target_rule(m, d):
            dep = dep_hour_D[d]
            return m.Ed[d, dep] + m.s_short[d] >= Etarget_D[d]
        m.DEPARTURE_TGT = pyo.Constraint(m.D, rule=departure_target_rule)

    
        m.SOC_ABS1 = pyo.Constraint(m.D, m.T, rule=lambda m,d,t: m.s_soc_abs[d,t] >= m.s_soc[d,t])
        m.SOC_ABS2 = pyo.Constraint(m.D, m.T, rule=lambda m,d,t: m.s_soc_abs[d,t] >= -m.s_soc[d,t])
        m.SOC_ABS_MAX = pyo.Constraint(m.D, m.T, rule=lambda m,d,t: m.s_soc_abs[d,t] <= S_SOC_MAX)




    # -------------------------
    # Network
    # ------------------------
    if include_network and USE_NETWORK:

        m.I = pyo.Set(initialize=I)
        m.L = pyo.Set(initialize=line_ids)

        # Fleet depot buses (only if fleet exists)
        if not no_fleet and fleet_mode == "opt":
            m.FB = pyo.Set(initialize=[18])
        else:
            m.FB = pyo.Set(initialize=[])

        m.Pload = pyo.Param(m.I, m.T, initialize=P_load_dict)
        m.Qload = pyo.Param(m.I, m.T, initialize=Qload_dict, mutable=False)
        m.PPV   = pyo.Param(m.I, m.T, initialize=PV_dict)

        SLACK_LIMIT_MW = 1 #slack limit represents transformer rating
        SLACK_LIMIT_MVA =1


        # Grid injection at GRID bus (pu)
        S_th = 1
        m.Pgrid = pyo.Var(m.T, bounds=(-S_th, S_th))
        m.Qgrid = pyo.Var(m.T, bounds=(-S_th, S_th))

        # in network section
        m.Pgrid_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Pgrid_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.PGRID_SPLIT = pyo.Constraint(m.T, rule=lambda m,t: m.Pgrid[t] == m.Pgrid_pos[t] - m.Pgrid_neg[t])
        
        # Grid import / export consistency constraints
        m.C_PGRID_POS1 = pyo.Constraint(m.T, rule=lambda m,t:
                                        m.Pgrid_pos[t] >= m.Pgrid[t]
                                    )

        # Pgrid_pos >= 0
        m.C_PGRID_POS2 = pyo.Constraint(m.T, rule=lambda m,t:
                                        m.Pgrid_pos[t] >= 0
                                    )

        # Pgrid_neg >= -Pgrid
        m.C_PGRID_NEG1 = pyo.Constraint(m.T, rule=lambda m,t:
                                        m.Pgrid_neg[t] >= -m.Pgrid[t]
                                    )

        # Pgrid_neg >= 0
        m.C_PGRID_NEG2 = pyo.Constraint(m.T, rule=lambda m,t:
                                        m.Pgrid_neg[t] >= 0
                                    )




        # Grid apparent power limit (linearized)
        m.GRID1 = pyo.Constraint(m.T,rule=lambda m,t:  m.Pgrid[t] + m.Qgrid[t] <= math.sqrt(2)*S_th)
        m.GRID2 = pyo.Constraint(m.T,rule=lambda m,t:  m.Pgrid[t] - m.Qgrid[t] <= math.sqrt(2)*S_th)
        m.GRID3 = pyo.Constraint(m.T,rule=lambda m,t: -m.Pgrid[t] + m.Qgrid[t] <= math.sqrt(2)*S_th)
        m.GRID4 = pyo.Constraint(m.T,rule=lambda m,t: -m.Pgrid[t] - m.Qgrid[t] <= math.sqrt(2)*S_th)


        S_slack = SLACK_LIMIT_MVA
        SQRT2 = math.sqrt(2.0)
    
        m.R    = pyo.Param(m.L, initialize=R_map)
        m.X    = pyo.Param(m.L, initialize=X_map)
        m.Smax = pyo.Param(m.L, initialize=Smax_map)

        m.Pij = pyo.Var(m.L, m.T)
        m.Qij = pyo.Var(m.L, m.T)
       
        # =========================
        # POWER BALANCE SLACK
        # # =========================
        m.ENS    = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

        m.PVcurt = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.Qens = pyo.Var(m.I, m.T, domain=pyo.Reals)
       
   

        m.ENS_LIM = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.ENS[i,t] <= m.Pload[i,t])
        m.PVC_LIM = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.PVcurt[i,t] <= m.PPV[i,t])
        m.QENS_LIM = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.Qens[i,t] <= abs(m.Qload[i,t]))
        m.QENS_LIM_POS = pyo.Constraint(m.I, m.T, rule=lambda m,i,t:  m.Qens[i,t] <= abs(m.Qload[i,t]))
        m.QENS_LIM_NEG = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: -m.Qens[i,t] <= abs(m.Qload[i,t]))


        m.xi_low  = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.xi_high = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)


        # =========================================================
        # Thermal violation
        # =========================================================

        m.sigma = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)


        # -----------------------------
        # Incremental voltage violation (relative to baseline)
        # -----------------------------
        m.VIOL = pyo.Expression(m.I, m.T, rule=lambda m,i,t: m.xi_low[i,t] + m.xi_high[i,t])


        # Reactive support vars
        m.QPV = pyo.Var(m.I, m.T)  # PV inverter Q
       
        def p_balance(m, i, t):
            inflow  = sum(m.Pij[l, t] for l in m.L if to_bus_map[l] == i)
            outflow = sum(m.Pij[l, t] for l in m.L if from_bus_map[l] == i)

            if not no_fleet:
                if fleet_mode == "opt":
                    fleet = m.Pd[18, t] if i == 18 else 0.0
                elif fleet_mode == "fixed":
                    fleet = m.Pfleet[t] if i == 18 else 0.0
                else:
                    fleet = 0.0
            else:
                fleet = 0.0

            Psrc    = m.Pgrid[t] if i == slack_bus else 0.0
            return (
                inflow - outflow
                + Psrc
                + (m.PPV[i, t] - m.PVcurt[i, t])
                - (m.Pload[i, t] - m.ENS[i, t])
                - fleet
                == 0
            )
       
        m.P_BAL = pyo.Constraint(m.I, m.T, rule=p_balance)
        
        def q_balance(m, i, t):
            inflow  = sum(m.Qij[l, t] for l in m.L if to_bus_map[l] == i)
            outflow = sum(m.Qij[l, t] for l in m.L if from_bus_map[l] == i)
            pv_q    = m.QPV[i, t]
           #fleet_q = m.Qfleet_bus[i, t] if (not no_fleet and i in m.FB) else 0.0
            Qsrc    = m.Qgrid[t] if i == slack_bus else 0.0
            return inflow - outflow + Qsrc + pv_q - m.Qload[i,t] + m.Qens[i,t] == 0
        # fleet_q == 0
        m.Q_BAL = pyo.Constraint(m.I, m.T, rule=q_balance)
       


        m.V = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.V_SLACK = pyo.Constraint(m.T,rule=lambda m,t: m.V[slack_bus,t] == 1.0)
        def vdrop_lin(m,l,t):
            f = from_bus_map[l]
            g = to_bus_map[l]
            return m.V[g,t] == m.V[f,t] - 2*(m.R[l]*m.Pij[l,t] + m.X[l]*m.Qij[l,t])
        m.VDROP = pyo.Constraint(m.L, m.T, rule=vdrop_lin)


       
        # Violation slacks measure distance outside limits
        m.VIOL_LOW  = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.xi_low[i,t]  >= V_MIN - m.V[i,t])
        m.VIOL_HIGH = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.xi_high[i,t] >= m.V[i,t] - V_MAX) # If inside bounds, optimizer will drive xi to zero (because you penalize xi in objective)



        # PV inverter PF limit (|Q| <= tanphi*(P - curt))
        m.QPV_CAP_POS = pyo.Constraint(m.I, m.T, rule=lambda m,i,t:  m.QPV[i,t] <= TANPHI*(m.PPV[i,t] - m.PVcurt[i,t]))
        m.QPV_CAP_NEG = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: -m.QPV[i,t] <= TANPHI*(m.PPV[i,t] - m.PVcurt[i,t]))


        # Linear thermal constraints (MILP-friendly)
        SQRT2 = math.sqrt(2.0)
        # Linearized apparent power limit (diamond approximation)
        m.THERM1 = pyo.Constraint(m.L, m.T, rule=lambda m,l,t:  m.Pij[l,t] + m.Qij[l,t] <= math.sqrt(2)*m.Smax[l] + m.sigma[l,t])
        m.THERM2 = pyo.Constraint(m.L, m.T,rule=lambda m,l,t:  m.Pij[l,t] - m.Qij[l,t] <= math.sqrt(2)*m.Smax[l] + m.sigma[l,t])
        m.THERM3 = pyo.Constraint(m.L, m.T,rule=lambda m,l,t: -m.Pij[l,t] + m.Qij[l,t] <= math.sqrt(2)*m.Smax[l] + m.sigma[l,t])
        m.THERM4 = pyo.Constraint(m.L, m.T,rule=lambda m,l,t: -m.Pij[l,t] - m.Qij[l,t] <= math.sqrt(2)*m.Smax[l] + m.sigma[l,t])



        # ================================
        # ENDOGENOUS CONGESTION METRIC
        # ================================
        # Total thermal overload at each hour
        m.SIGMA_T = pyo.Expression(m.T, rule=lambda m,t:
            sum(m.sigma[l,t] for l in m.L)
        )

        # Total voltage violation magnitude
        m.VIOL_T = pyo.Expression(m.T, rule=lambda m,t:
            sum(m.xi_low[i,t] + m.xi_high[i,t] for i in m.I)
        )

        # Combined congestion severity index
        W_ENS = 5.0   # weight to reflect severits

        m.CONG_TH_T = pyo.Expression(m.T,rule=lambda m,t: sum(m.sigma[l,t] / (m.Smax[l] + EPS_CONG) for l in m.L))
        m.CONG_V_T = pyo.Expression(m.T,rule=lambda m,t: sum((m.xi_low[i,t] + m.xi_high[i,t]) / DV_BAND for i in m.I))
        m.CONG_RAW_T = pyo.Expression(m.T,rule=lambda m,t: W_TH * m.CONG_TH_T[t] + W_V * m.CONG_V_T[t])
        m.CONG_ABS_T = pyo.Expression(m.T,rule=lambda m,t: m.CONG_RAW_T[t] / gamma_value)
        #m.CONG_INC_T = pyo.Expression(m.T,rule=lambda m,t: m.CONG_ABS_T[t] - m.CONG_REF[t])
        m.CONG_INC_T = pyo.Expression(m.T, rule=lambda m,t: m.CONG_RAW_T[t] - m.CONG_REF[t])

        Mlad = 1.0 if bigm_ladder is None else float(bigm_ladder)
        
       



        # -----------------------------
        # TARIFF LADDER (ONLY if enabled)
        # -----------------------------
        if tariff_logic:
            # Tariff step levels chosen by DSO
            #m.K = pyo.RangeSet(0, len(THETA)-2)   # e.g. 0..3 if 4 thresholds
            m.K = pyo.RangeSet(0, len(theta)-2)


            # Fixed DUoS tariff ladder (€/MWh) PUTA 10 ZBOG KONVERZIJE U PU
            lambda_levels = {
                0:  0.0,   # no congestion component
                1: 200.0,   # mild congestion
                2: 400.0,   # moderate
                3: 800.0,   # severe
                4: 1400.0,  # near‑critical
            }


        
            m.lambda_k = pyo.Param(m.K, initialize=lambda_levels, mutable=False)
            # Optional: upper bound on tariff levels
            #Mlad = BIGM_LADDER if bigm_ladder is None else bigm_ladder
            MAX_CONG_T = sum( (1.0 * 1.0) / (Smax_map[l] + EPS_CONG) for l in line_ids ) \
                + sum( (1.0 * DV_BAND) / DV_BAND for i in I )   # = len(I)
            
            


            m.z = pyo.Var(m.K, m.T, domain=pyo.Binary)
            m.ONE_STEP = pyo.Constraint(m.T, rule=lambda m,t: sum(m.z[k,t] for k in m.K) == 1)
            # Auxiliary variable for product lambda_k * z[k,t]
            m.lambda_step = pyo.Var(m.K, m.T, domain=pyo.NonNegativeReals)

            # BIG M CONSTRAINTS
            M_LAMBDA = LAMBDA_MAX

            # Linearization of lambda_step = lambda_k * z
    
            ## lambda_step <= lambda_k
            m.LSTEP1 = pyo.Constraint(m.K, m.T,  rule=lambda m,k,t: m.lambda_step[k,t] <= m.lambda_k[k])
            
            # lambda_step <= M * z
            m.LSTEP2 = pyo.Constraint(m.K, m.T, rule=lambda m,k,t: m.lambda_step[k,t] <= M_LAMBDA * m.z[k,t])
            
            # lambda_step >= lambda_k - M*(1 - z)
            m.LSTEP3 = pyo.Constraint(m.K, m.T,rule=lambda m,k,t: m.lambda_step[k,t] >= m.lambda_k[k] - M_LAMBDA*(1 - m.z[k,t]))
            
            # lambda_step >= 0  (already enforced by domain, but explicit if desired)
            m.LSTEP4 = pyo.Constraint(m.K, m.T,rule=lambda m,k,t: m.lambda_step[k,t] >= 0)



            # lambda definition
            m.LAMBDA_DEF = pyo.Constraint(m.T,rule=lambda m,t: m.lambda_tar[t] == sum(m.lambda_step[k,t] for k in m.K))

            def lower_threshold_rule(m, k, t):
                if k == 0:
                    return pyo.Constraint.Skip
                return m.CONG_INC_T[t] >= theta[k] - Mlad*(1 - m.z[k,t])
            
            def upper_threshold_rule(m, k, t):
                if k == m.K.last():
                    return pyo.Constraint.Skip
                return m.CONG_INC_T[t] <= theta[k+1] + Mlad*(1 - m.z[k,t])
           
            m.LADDER_LOW = pyo.Constraint(m.K, m.T, rule=lower_threshold_rule)
            m.LADDER_UP  = pyo.Constraint(m.K, m.T, rule=upper_threshold_rule)




        # ------------------------------------------------
        # TARIFF LADDER LOGIC
        # -------------------------------------------------
        # ==========================================
        # # Marginal congestion tariff (true DLMP)
        # # ==========================================
           
        if (not no_fleet) and (fixed_fleet_profile is None):
            m.LL_OBJ = pyo.Objective(
                expr=
                sum((m.price[t] + m.lambda_tar[t]) * m.Pd[d, t] * DELTA_T
                    for d in m.D for t in m.T)
                    + C_SHORT * sum(m.s_short[d] for d in m.D),
                    #+ SOC_SLACK_PEN * sum(m.s_soc_abs[d, t] for d in m.D for t in m.T_NOT0),
                    sense=pyo.minimize
                )
            if use_kkt:
                m.LL_OBJ.deactivate()
 

        if use_kkt:   
            # DUAL VARIABLES
            m.muP_min = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
            m.muP_max = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
            
            # E_t is fixed at t0, so KKT only on T_NOT0
            m.muE_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.muE_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            
            m.pi_dep  = pyo.Var(m.D, domain=pyo.NonNegativeReals)
            m.nu = pyo.Var(m.D, m.T, domain=pyo.Reals)
            
            # Missing lower-bound dual for s_short >= 0
            m.muShort_min = pyo.Var(m.D, domain=pyo.NonNegativeReals)
            
            # Duals for s_soc bounds
            m.muSoc_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)   # s_soc >= -S_SOC_MAX
            m.muSoc_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)   # s_soc <=  S_SOC_MAX
            
            # Duals for s_soc_abs bounds
            m.muAbs_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)   # s_soc_abs >= 0
            m.muAbs_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)   # s_soc_abs <= S_SOC_MAX
            
            # Duals for |s_soc| linearization
            m.alpha_abs1 = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)  # s_soc - s_soc_abs <= 0
            m.alpha_abs2 = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)  # -s_soc - s_soc_abs <= 0
            
            # BINARY VARIABLES FOR BIG-M COMPLEMENTARITY

            m.yP_min = pyo.Var(m.D, m.T, within=pyo.Binary)
            m.yP_max = pyo.Var(m.D, m.T, within=pyo.Binary)
            
            m.yE_min = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.yE_max = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            
            m.yDEP   = pyo.Var(m.D, within=pyo.Binary)
            m.yShort_min = pyo.Var(m.D, within=pyo.Binary)
            
            m.ySoc_min = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.ySoc_max = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.yAbs_min = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.yAbs_max = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.yAbs1 = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            m.yAbs2 = pyo.Var(m.D, m.T_NOT0, within=pyo.Binary)
            
            # SLACK UPPER BOUNDS
            M_P_dt = {(d,t): float(value(m.U_D[d,t])) for d in m.D for t in m.T}
            M_E_d = {d: float(Emax_D[d]) for d in m.D}
            M_SHORT_d = {d: float(Emax_D[d]) for d in m.D}
            M_SOC = 2.0 * S_SOC_MAX
            M_ABS = S_SOC_MAX
            M_ABS_REL = 2.0 * S_SOC_MAX
            M_DUAL = BIGM
            
            # SLACK VARIABLES 
            m.sP_min = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)
            m.sP_max = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)          
            m.sE_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.sE_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            
            m.sDEP = pyo.Var(m.D, domain=pyo.NonNegativeReals)
            m.sShortLB = pyo.Var(m.D, domain=pyo.NonNegativeReals)
            
            m.sSoc_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.sSoc_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            
            m.sAbs_min = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.sAbs_max = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.sAbs1_slack = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            m.sAbs2_slack = pyo.Var(m.D, m.T_NOT0, domain=pyo.NonNegativeReals)
            
            # SLACK DEFINITIONS
            m.DEF_sP_min = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.sP_min[d,t] == m.Pd[d,t])
            m.DEF_sP_max = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.sP_max[d,t] == m.U_D[d,t] - m.Pd[d,t])
            m.DEF_sE_min = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sE_min[d,t] == m.Ed[d,t])
            m.DEF_sE_max = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sE_max[d,t] == Emax_D[d] - m.Ed[d,t])
            m.DEF_sDEP = pyo.Constraint(m.D,rule=lambda m,d: m.sDEP[d] == m.Ed[d, dep_hour_D[d]] + m.s_short[d] - Etarget_D[d])
            m.DEF_sShortLB = pyo.Constraint(m.D,rule=lambda m,d: m.sShortLB[d] == m.s_short[d])
            m.DEF_sSoc_min = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sSoc_min[d,t] == m.s_soc[d,t] + S_SOC_MAX)
            m.DEF_sSoc_max = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sSoc_max[d,t] == S_SOC_MAX - m.s_soc[d,t])
            m.DEF_sAbs_min = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs_min[d,t] == m.s_soc_abs[d,t])
            m.DEF_sAbs_max = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs_max[d,t] == S_SOC_MAX - m.s_soc_abs[d,t])
            m.DEF_sAbs1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs1_slack[d,t] == m.s_soc_abs[d,t] - m.s_soc[d,t])
            m.DEF_sAbs2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs2_slack[d,t] == m.s_soc_abs[d,t] + m.s_soc[d,t])
            
            
            # BIG-M COMPLEMENTARITY
            m.COMP_PMIN1 = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.muP_min[d,t] <= M_DUAL * m.yP_min[d,t])
            m.COMP_PMIN2 = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.sP_min[d,t] <= M_P_dt[(d,t)] * (1 - m.yP_min[d,t]))
            m.COMP_PMAX1 = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.muP_max[d,t] <= M_DUAL * m.yP_max[d,t])
            m.COMP_PMAX2 = pyo.Constraint(m.D, m.T,rule=lambda m,d,t: m.sP_max[d,t] <= M_P_dt[(d,t)] * (1 - m.yP_max[d,t]))
            m.COMP_EMIN1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muE_min[d,t] <= M_DUAL * m.yE_min[d,t])
            m.COMP_EMIN2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sE_min[d,t] <= M_E_d[d] * (1 - m.yE_min[d,t]))
            m.COMP_EMAX1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muE_max[d,t] <= M_DUAL * m.yE_max[d,t])
            m.COMP_EMAX2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sE_max[d,t] <= M_E_d[d] * (1 - m.yE_max[d,t]))
            m.COMP_DEP1 = pyo.Constraint(m.D,rule=lambda m,d: m.pi_dep[d] <= M_DUAL * m.yDEP[d])
            m.COMP_DEP2 = pyo.Constraint(m.D,rule=lambda m,d: m.sDEP[d] <= M_E_d[d] * (1 - m.yDEP[d]))
            m.COMP_SHORT1 = pyo.Constraint(m.D,rule=lambda m,d: m.muShort_min[d] <= M_DUAL * m.yShort_min[d])
            m.COMP_SHORT2 = pyo.Constraint(m.D,rule=lambda m,d: m.sShortLB[d] <= M_SHORT_d[d] * (1 - m.yShort_min[d]))
            m.COMP_SOCMIN1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muSoc_min[d,t] <= M_DUAL * m.ySoc_min[d,t])
            m.COMP_SOCMIN2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sSoc_min[d,t] <= M_SOC * (1 - m.ySoc_min[d,t]))
            m.COMP_SOCMAX1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muSoc_max[d,t] <= M_DUAL * m.ySoc_max[d,t])
            m.COMP_SOCMAX2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sSoc_max[d,t] <= M_SOC * (1 - m.ySoc_max[d,t]))
            m.COMP_ABSMIN1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muAbs_min[d,t] <= M_DUAL * m.yAbs_min[d,t])
            m.COMP_ABSMIN2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs_min[d,t] <= M_ABS * (1 - m.yAbs_min[d,t]))
            m.COMP_ABSMAX1 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.muAbs_max[d,t] <= M_DUAL * m.yAbs_max[d,t])
            m.COMP_ABSMAX2 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs_max[d,t] <= M_ABS * (1 - m.yAbs_max[d,t]))
            m.COMP_ABS11 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.alpha_abs1[d,t] <= M_DUAL * m.yAbs1[d,t])
            m.COMP_ABS12 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs1_slack[d,t] <= M_ABS_REL * (1 - m.yAbs1[d,t]))
            m.COMP_ABS21 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.alpha_abs2[d,t] <= M_DUAL * m.yAbs2[d,t])
            m.COMP_ABS22 = pyo.Constraint(m.D, m.T_NOT0,rule=lambda m,d,t: m.sAbs2_slack[d,t] <= M_ABS_REL * (1 - m.yAbs2[d,t]))
            
            # EXACT STATIONARITY
            m.ST_P = pyo.Constraint(m.D, m.T,rule=lambda m,d,t:(m.price[t] + m.lambda_tar[t]) * DELTA_T - eta_D[d] * DELTA_T * m.nu[d,t] + m.muP_max[d,t] - m.muP_min[d,t]  == 0)
            def st_e_rule(m, d, t):
                if t == m.T.first():
                    return pyo.Constraint.Skip
                nu_next = 0.0 if t == m.T.last() else m.nu[d, m.T.next(t)]
                dep_term = -m.pi_dep[d] if t == dep_hour_D[d] else 0.0
                return m.nu[d,t] - nu_next + m.muE_max[d,t] - m.muE_min[d,t] + dep_term == 0
            m.ST_E = pyo.Constraint(m.D, m.T, rule=st_e_rule)

            m.ST_SHORT = pyo.Constraint(m.D,rule=lambda m,d: C_SHORT - m.pi_dep[d] - m.muShort_min[d] == 0)
            
            m.ST_SSOC = pyo.Constraint(m.D, m.T_NOT0,
                                       rule=lambda m,d,t:
                                       -m.nu[d,t]
                                       + m.alpha_abs1[d,t] - m.alpha_abs2[d,t]
                                       + m.muSoc_max[d,t] - m.muSoc_min[d,t]
                                       == 0
                                    )

            m.ST_SSOCABS = pyo.Constraint(m.D, m.T_NOT0,
                                          rule=lambda m,d,t:
                                          SOC_SLACK_PEN
                                          - m.alpha_abs1[d,t] - m.alpha_abs2[d,t]
                                          + m.muAbs_max[d,t] - m.muAbs_min[d,t]
                                          == 0
                                        )
           
    # =========================================================
    # DSO OBJECTIVE — ALIGNED WITH CONG_T
    # =========================================================
    if include_network and USE_NETWORK:



        net_cost = (
            #C_ENS * sum(m.ENS_pos[i,t] * DELTA_T for i in m.I for t in m.T)
            + C_PVC * sum(m.PVcurt[i,t] * DELTA_T for i in m.I for t in m.T)
            + C_VOLT * sum(m.xi_low[i,t] + m.xi_high[i,t] for i in m.I for t in m.T)
            + C_THERM * sum(m.sigma[l,t] for l in m.L for t in m.T)
        
        )
    else:
        net_cost = 0.0


    # reference tariff (optional)
    lambda_ref = {t: 0.0 for t in T}
    m.lambda_ref = pyo.Param(m.T, initialize=lambda_ref)
    m.dev_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.dev_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.DEV_SPLIT = pyo.Constraint(m.T,rule=lambda m,t:m.lambda_tar[t] - m.lambda_ref[t] == m.dev_pos[t] - m.dev_neg[t])

    # Linearization of lambda * Pfleet using McCormick envelopes


    # Upper bound of fleet charging in pu
    PFLEET_MAX = max(value(m.U_D[d,t]) for d in m.D for t in m.T) if fleet_mode == "opt" else float(Pmax_fleet) * (fleet_scale if fleet_scale is not None else 1.0)

    # Bounds of variables
    Pmin = 0.0
    Pmax = PFLEET_MAX

    Lmin = 0.0
    Lmax = LAMBDA_MAX

    # Auxiliary variable for lambda * Pfleet
    m.w_duos = pyo.Var(m.T)

    # McCormick envelopes
    def mcc1(m, t):
        return m.w_duos[t] >= Lmin * m.Pfleet[t] + Pmin * m.lambda_tar[t] - Lmin * Pmin
    m.MCC1 = pyo.Constraint(m.T, rule=mcc1)

    def mcc2(m, t):
        return m.w_duos[t] >= Lmax * m.Pfleet[t] + Pmax * m.lambda_tar[t] - Lmax * Pmax
    m.MCC2 = pyo.Constraint(m.T, rule=mcc2)

    def mcc3(m, t):
        return m.w_duos[t] <= Lmax * m.Pfleet[t] + Pmin * m.lambda_tar[t] - Lmax * Pmin
    m.MCC3 = pyo.Constraint(m.T, rule=mcc3)
    
    def mcc4(m, t):
        return m.w_duos[t] <= Lmin * m.Pfleet[t] + Pmax * m.lambda_tar[t] - Lmin * Pmax
    m.MCC4 = pyo.Constraint(m.T, rule=mcc4)


    # Fleet cost calculation


    fleet_energy_payment = sum(m.price[t] * (m.Pfleet[t] ) * DELTA_T for t in m.T)

    fleet_duos_payment = sum(m.w_duos[t] * DELTA_T for t in m.T)
    fleet_cost_expression = fleet_energy_payment + fleet_duos_payment

    # =========================================================
    # OBJECTIVE (UPPER LEVEL)
    # =========================================================

    m.OBJ = pyo.Objective(
        expr =
        C_VOLT * sum(m.xi_low[i,t] + m.xi_high[i,t] for i in m.I for t in m.T)
        + C_THERM * sum(m.sigma[l,t] for l in m.L for t in m.T)
        + C_ENS  * sum(m.ENS[i,t] for i in m.I for t in m.T)
        + C_PVC  * sum(m.PVcurt[i,t] for i in m.I for t in m.T),
       # + C_REG * sum(m.dev_pos[t] + m.dev_neg[t] for t in m.T),    
        sense=pyo.minimize
    )

    return m

def activate_only(model, name):
    # Deactivate all objectives
    for obj in model.component_objects(pyo.Objective, active=True):
        obj.deactivate()
        # Activate the one you want
    getattr(model, name).activate()



def safe_value(x, default=0.0):

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    try:
        v = value(x, exception=False)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


#    Returns physical grid metrics:
#      - total voltage violation (sum xi_low + xi_high)
#      - total thermal slack (sum sigma)
#      - total PV curtailment (energy)
#      - total ENS (energy)
def compute_physical_metrics(m):
    total_voltage_violation = sum(
        value(m.xi_low[i,t]) + value(m.xi_high[i,t])
        for i in m.I for t in m.T
    )

    total_thermal_slack = sum(
        value(m.sigma[l,t])
        for l in m.L for t in m.T
    )

    total_pv_curt = sum(
        value(m.PVcurt[i,t]) * DELTA_T
        for i in m.I for t in m.T
    )

    total_ens = sum(
        value(m.ENS[i,t]) * DELTA_T
        for i in m.I for t in m.T
    )

    return (
        float(total_voltage_violation),
        float(total_thermal_slack),
        float(total_pv_curt),
        float(total_ens)
    )


#    Prints:
#      - min_V, max_V
#      - total voltage violation sum(xi_low+xi_high) over all buses+hours
#      - total thermal slack sum(sigma) over all lines+hours
#      - total PV curtailment and hourly system PV curtailment profile

def print_network_diagnostics(m, label=""):

    # Voltage extrema
    min_V = min(safe_value(m.V[i,t]) for i in m.I for t in m.T)
    max_V = max(safe_value(m.V[i,t]) for i in m.I for t in m.T)

    # Total violations / slacks
    xi_total = sum(safe_value(m.xi_low[i,t]) + safe_value(m.xi_high[i,t]) for i in m.I for t in m.T)
    sigma_total = sum(safe_value(m.sigma[l,t]) for l in m.L for t in m.T)

    # PV curtailment: total + hourly (system sum)
    pv_curt_total = sum(safe_value(m.PVcurt[i,t]) * DELTA_T for i in m.I for t in m.T)
    pv_curt_hourly = {t: sum(safe_value(m.PVcurt[i,t]) for i in m.I) for t in m.T}

    print(f"\n===== NETWORK DIAGNOSTICS {label} =====")
    print(f"Voltage min/max (V^2 pu): {min_V:.6f} / {max_V:.6f}   (limits: {V_MIN:.6f} .. {V_MAX:.6f})")
    print(f"Total voltage violation sum(xi_low+xi_high) [pu^2]: {xi_total:.6f}")
    print(f"Total thermal slack sum(sigma) [pu]: {sigma_total:.6f}")
    print(f"Total PV curtailment [pu*hour]: {pv_curt_total:.6f}")


    # Print hours where curtailment occurs
    hours_with_curt = [(t, pv_curt_hourly[t]) for t in m.T if pv_curt_hourly[t] > 1e-9]
    if len(hours_with_curt) == 0:
        print("PV curtailment occurs in: none")
    else:
        print("PV curtailment occurs hourly (system total PVcurt):")
        for t, v in hours_with_curt:
            print(f"  t={int(t):02d}: {v:.6f} pu")

    # Optional: show top buses contributing to curtailment (total over day)
    pv_curt_by_bus = {i: sum(safe_value(m.PVcurt[i,t]) * DELTA_T for t in m.T) for i in m.I}
    top = sorted(pv_curt_by_bus.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top = [(i, e) for (i, e) in top if e > 1e-9]
    if top:
        print("Top PV-curtailing buses (energy over day):")
        for i, e in top:
            print(f"  bus {int(i):>2}: {e:.6f} pu*hour")

# ======================================================================
# SECTION 2.5 — EXPERIMENT RUNNER
# ======================================================================
def solve_for_fleet_size(fleet_scale):

    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"] = 0.001
    solver.options["TimeLimit"] = 1100
    hourly = {}


    def compute_metrics(model):
        sigma_t = np.array([
            sum(value(model.sigma[l,t])  for l in model.L)
            for t in model.T
        ])
       
        xi_t = np.array([
            sum(value(model.xi_low[i,t]) + value(model.xi_high[i,t]) for i in model.I)
            for t in model.T
        ])
       
        #cong_t = np.array([value(model.CONG_ABS_T[t]) for t in model.T])
        cong_t = np.array([value(model.CONG_ABS_T[t]) for t in model.T])
        
               
        pv_curt = sum(value(model.PVcurt[i,t])  * DELTA_T for i in model.I for t in model.T)
       
        ens = sum( value(model.ENS[i,t]) * DELTA_T for i in model.I for t in model.T)
        return cong_t, float(np.sum(cong_t)), pv_curt, ens
   
    def safe_dual(model, constr):
        try:
            if not hasattr(model, "dual"):
                return 0.0
            v = model.dual.get(constr, None)
            return 0.0 if v is None else float(value(v))
        except Exception:
            return 0.0

    # =====================================================
    # CASE 0 — BASE GRID (NO FLEET)
    # =====================================================

    m0 = build_model(
        include_network=True,
        use_kkt=False,
        tariff_logic=False,
        fleet_mode="none"
    )
    activate_only(m0, "OBJ")
    solver.solve(m0)
    #cong_ref_t = {t: value(m0.CONG_ABS_T[t]) for t in m0.T}
    cong_ref_t = {t: value(m0.CONG_RAW_T[t]) for t in m0.T}


    volt0, therm0, pvc0, ens0 = compute_physical_metrics(m0)
   
    print("\n--- CASE 0 (No Fleet) ---")
    print("Total voltage violation :", volt0)
    print("Total thermal slack     :", therm0)
    print("Total PV curtailment    :", pvc0)
    print("Total ENS               :", ens0)


   
    # -------------------------------------------------
    # # DLMP reference from baseline OPF (CASE 0 only)
    # # -------------------------------------------------
    dlmp_bus = {}
    dlmp_avg = {}
    if len(m0.dual) == 0:
        print("No duals available for baseline.")
    else:
        for t in m0.T:
            vals = []
            for i in fleet_buses:
                if (i, t) in m0.P_BAL:
                    try:
                        dual_val = m0.dual[m0.P_BAL[i,t]]
                        if dual_val is None:
                            dual_val = 0.0
                    except KeyError:
                        dual_val = 0.0
                    dlmp_pu = - value(dual_val)
                    dlmp_eur = dlmp_pu / S_BASE_MVA
                    vals.append(dlmp_eur)
            dlmp_avg[t] = float(np.mean(vals)) if len(vals) > 0 else 0.0  

   
 
    cong0_t = np.array([value(m0.CONG_ABS_T[t]) for t in m0.T])
    cong0_total = float(np.sum(cong0_t))



    # CASE F — Grid + fleet uncontrolled (fixed ASAP profile), no tariff
    Pfleet_asap = build_uncontrolled_profile(fleet_scale=fleet_scale)   # returns {t: Pfleet}

    mF = build_model(
        include_network=True,
        use_kkt=False,
        tariff_logic=False,
        fleet_scale=fleet_scale,
        fleet_mode="fixed",
        fixed_fleet_profile=Pfleet_asap
       
    )
    #cong_ref_t = {t: value(mF.CONG_ABS_T[t]) for t in mF.T}

   



    activate_only(mF, "OBJ")
    resultsF = solver.solve(mF)
    congF = np.array([value(mF.CONG_RAW_T[t]) for t in mF.T], dtype=float)
    cong0 = np.array([value(m0.CONG_RAW_T[t]) for t in m0.T], dtype=float)
    incF = np.maximum(0.0, congF - cong0)
    active = incF[incF > 1e-6]


    term = resultsF.solver.termination_condition
    if term not in [TerminationCondition.optimal,
                    TerminationCondition.feasible,
                    TerminationCondition.maxTimeLimit]:
        print("Case F infeasible:", term)
        return None
    
    congF_abs_t = np.array([value(mF.CONG_ABS_T[t]) for t in mF.T]) 
    
    charging_caseF_array = np.array([value(mF.Pfleet[t]) for t in mF.T], dtype=float)
    pv_caseF_array = np.array([sum(PV_dict[(i, t)] - value(mF.PVcurt[i, t]) for i in mF.I) for t in mF.T], dtype=float)
    load_caseF_array = np.array([sum(value(mF.Pload[i, t]) for i in mF.I) + value(mF.Pfleet[t]) for t in mF.T], dtype=float)
    thermal_caseF_array = np.array([sum(value(mF.sigma[l, t]) for l in mF.L) for t in mF.T], dtype=float)
    voltage_caseF_array = np.array([sum(value(mF.xi_low[i, t]) + value(mF.xi_high[i, t]) for i in mF.I) for t in mF.T], dtype=float)


    # =====================================================
    # CASE 0.1 — FLEET, NO TARIFF (λ = 0)
    # =====================================================


    m_base = build_model(
        include_network=True,
        use_kkt=True,
        tariff_logic=False,
        fleet_scale=fleet_scale,
        no_fleet=False
    )
           
    activate_only(m_base, "OBJ")

   
    solver.options["MIPGap"] = 0.001
    solver.options["TimeLimit"] = 1100
    solver.options["MIPFocus"] = 1
    solver.options["Heuristics"] = 0.5
    solver.options["PumpPasses"] = 50
    solver.options["RINS"] = 50
    solver.options["Cuts"] = 2
    solver.options["Presolve"] = 2
    
    results_base = solver.solve(m_base, tee=True)


    if results_base.solver.termination_condition in [
        TerminationCondition.optimal,
        TerminationCondition.maxTimeLimit]:
        print("Solved")
    
    print("Baseline total fleet energy:", sum(value(m_base.Pfleet[t]) for t in m_base.T)*DELTA_T)
    print("Baseline total shortfall:", sum(value(m_base.s_short[d]) for d in m_base.D))

    
    # VALIDATION — BASELINE (Fleet, No Tariff)
    print("\n=== VALIDATION: BASELINE ===")
    
    # KKT residual check

    if use_kkt:
        def safe_val(v):
            return value(v) if v.value is not None else float("nan")
        
        max_st_p = max(abs(value(m_base.ST_P[d,t].body())) for d in m_base.D for t in m_base.T)
        max_st_e = max(abs(value(m_base.ST_E[d,t].body())) for d in m_base.D for t in m_base.T if t != m_base.T.first())
        max_st_s = max(abs(value(m_base.ST_SHORT[d].body())) for d in m_base.D)
        print("Max |ST_P|:", max_st_p)
        print("Max |ST_E|:", max_st_e)
        print("Max |ST_SHORT|:", max_st_s)

    # -----------------------------------------------------
    # Incremental congestion caused by fleet
    # (relative to Case 0 — no fleet)
    ## - ----------------------------------------------------
    sigma_base_t = np.array([sum(value(m_base.sigma[l, t]) for l in m_base.L) for t in m_base.T])
    xi_base_t = np.array([sum(value(m_base.xi_low[i, t]) + value(m_base.xi_high[i, t])for i in m_base.I)for t in m_base.T])

    #FOR RQ2
    # # PV curtailment baseline
    pv_curt_base = sum(value(m_base.PVcurt[i,t]) * DELTA_T for i in m_base.I for t in m_base.T)
    # Unmet charging demand baseline
    unmet_energy_base = sum(value(m_base.s_short[d]) for d in m_base.D)
    
    
    # Congestion baseline
    cong_base_total = sum(value(m_base.CONG_ABS_T[t]) for t in m_base.T)
    
    # Baseline objective value (DSO objective)
    OBJ_base = value(m_base.OBJ)

    # Baseline charging profile
    charging_base_array = np.array([value(m_base.Pfleet[t]) for t in m_base.T])

   
    # -----------------------------------------------------
    # Baseline ENS & PV Curtailment
    ## -----------------------------------------------------
    ens_base_t = np.array([sum(value(m_base.ENS[i, t]) for i in m_base.I)for t in m_base.T])
    ens_base_total = float(np.sum(ens_base_t) * DELTA_T)
    pv_curt_base = float(sum(value(m_base.PVcurt[i, t]) * DELTA_T for i in m_base.I for t in m_base.T))
    

    # -----------------------------------------------------
    # Baseline fleet charging profile
    # -----------------------------------------------------
    
    charging_base_array = np.array([value(m_base.Pfleet[t]) for t in m_base.T])
        
    charging_base = {t: value(m_base.Pfleet[t]) for t in m_base.T}

   
    # -----------------------------------------------------
    # Voltage diagnostics
    # -----------------------------------------------------
    print_network_diagnostics(m_base, label=f"(CASE 0.1 baseline, fleet_scale={fleet_scale})")

       

    # =====================================================
    # CASE 1 — FLEET + TARIFF
    # =====================================================
    # Build congestion ladder based on fleet-caused congestion
    baseline_peak = max(congF_abs_t)
    eps = 1e-6


    MAX_SIGMA = len(line_ids) * max(Smax_map.values())
    MAX_XI    = len(I) * DV_BAND
    MAX_CONG  = (W_TH * MAX_SIGMA + W_V * MAX_XI)
    BIGM_LADDER = MAX_CONG

    rawF = np.array([value(mF.CONG_RAW_T[t]) for t in mF.T], dtype=float)
    gamma_value = max(1e-6, np.percentile(rawF, 95))
    
    def compute_bigM_ladder():
        MAX_SIGMA = len(line_ids) * max(Smax_map.values())
        MAX_XI    = len(I) * DV_BAND
        return (W_TH * MAX_SIGMA + W_V * MAX_XI)
    

    M_sigma = max(Smax_map.values())
    M_xi    = DV_BAND

    if len(active) < 4:
        mx = max(active) if len(active) > 0 else 1e-4
        theta_dynamic = [0.0, 0.2*mx, 0.4*mx, 0.7*mx, 0.9*mx, 1.01*mx]
    else:
        q1, q2, q3, q4 = np.quantile(active, [0.2, 0.4, 0.7, 0.9])
        eps = max(1e-6, 0.01 * max(active))
        theta_dynamic = [
            0.0,
            max(eps, q1),
            max(q1 + eps, q2),
            max(q2 + eps, q3),
            max(q3 + eps, q4),
            max(q4 + eps, active.max() + eps),
        ]
    
    # intervals:
    # [0.00, 0.05)  -> step 0 (no congestion)
    # [0.05, 0.15)  -> step 1 (mild)
    # [0.15, 0.35)  -> step 2 (moderate)
    # [0.35, 0.70)  -> step 3 (severe)
    # [0.70, 1.00]  -> step 4 (near‑critical)


    MAX_CONG_T = sum( (1.0 * 1.0) / (Smax_map[l] + EPS_CONG) for l in line_ids ) \
        + sum( (1.0 * DV_BAND) / DV_BAND for i in I )   # = len(I)
    
    Mlad = max(1.0, max(theta_dynamic) - min(theta_dynamic) + 0.1)
   

    m = build_model(
        include_network=True,
        use_kkt=True,
        tariff_logic=True,        
        fleet_scale=fleet_scale,
        no_fleet=False,
        theta_levels=theta_dynamic,
        bigm_ladder=Mlad,
        gamma_value=gamma_value,
        cong_ref=cong_ref_t
        #cong_ref=cong_ref_t_caseF
  
    )

    print("Using thresholds:", theta_dynamic)

    solver.options["MIPFocus"] = 1        # focus on finding feasible incumbents
    solver.options["Heuristics"] = 0.5    # more primal heuristics
    solver.options["PumpPasses"] = 50     # feasibility pump
    solver.options["RINS"] = 50           # more large-neighborhood search
    solver.options["Cuts"] = 2            # moderate cuts
    solver.options["Presolve"] = 2
    print("\n=== SCALE CHECK ===")
    print("theta_dynamic:", theta_dynamic)
    print("max Case0 raw congestion:", max(value(m0.CONG_RAW_T[t]) for t in m0.T))
    print("max CaseF raw congestion:", max(value(mF.CONG_RAW_T[t]) for t in mF.T))
    print("gamma_value:", gamma_value)
    activate_only(m, "OBJ")
    results_tar = solver.solve(m, tee=True)

    print("\n=== Tariff ladder diagnostics ===")
    for t in m.T:
        cong_abs = value(m.CONG_ABS_T[t])
        cong_ref = value(m.CONG_REF[t])
        cong_inc = value(m.CONG_INC_T[t])
        lam = value(m.lambda_tar[t])
        zvals = {k: value(m.z[k,t]) for k in m.K}
        active_k = [k for k,v in zvals.items() if v > 0.5]
        print(
            f"t={t:02d}  "
            f"CONG_ABS={cong_abs:.6f}  "
            f"CONG_REF={cong_ref:.6f}  "
            f"CONG_INC={cong_inc:.6f}  "
            f"lambda={lam:.1f}  step={active_k}"
        )
    print("\n=== TARIFF RAW DIAGNOSTICS ===")
    for t in m.T:
        print(
            f"t={t:02d}  "
            f"RAW={value(m.CONG_RAW_T[t]):.6f}  "
            f"REF={value(m.CONG_REF[t]):.6f}  "
            f"INC={value(m.CONG_INC_T[t]):.6f}  "
            f"lambda={value(m.lambda_tar[t]):.1f}"
        )

    
    term = results_tar.solver.termination_condition

    cong_tar_abs_t = np.array([value(m.CONG_ABS_T[t]) for t in m.T])
    lambda_t = np.array([value(m.lambda_tar[t]) for t in m.T])
    print("Congestion min/max:", cong_tar_abs_t.min(), cong_tar_abs_t.max())
    print("Lambda min/max:", lambda_t.min(), lambda_t.max())
    print("Tariff total fleet energy:",   sum(value(m.Pfleet[t]) for t in m.T)*DELTA_T)
    print("Tariff total shortfall:", sum(value(m.s_short[d]) for d in m.D))
    print("Total fleet energy MWh:",
    sum(value(m.Pfleet[t]) for t in m.T) * S_BASE_MVA)

    base_load_only = np.array([sum(P_load_dict[i, t] for i in I) for t in m.T], dtype=float)
    
    
    price_array = np.array([value(m.price[t]) for t in m.T])
    avg_price = np.mean(price_array)
    print("Average energy price €/MWh:", avg_price)
    energy_MWh = sum(value(m.Pfleet[t]) for t in m.T) * S_BASE_MVA
    print("Expected daily cost approx:",
          energy_MWh * avg_price)
    
    max_soc_slack = max(
        abs(safe_value(m.s_soc[d,t]))
        for d in m.D for t in m.T
    )
    print("Max SOC slack:", max_soc_slack)
   
    for t in m.T:
        if value(m.CONG_INC_T[t]) > theta_dynamic[1]:
            print("Should activate at hour", t)

    cong_hours = [t for t in m.T if value(m.CONG_INC_T[t]) > theta_dynamic[1]]
    print("Congested hours:", cong_hours)

    
    charging_tariff = {
        t: value(m.Pfleet[t])
        for t in m.T
    }

    print("Congestion min/max:",
          min(value(m.CONG_ABS_T[t]) for t in m.T),
          max(value(m.CONG_ABS_T[t]) for t in m.T))
    
    
    print("CONG_ABS min/max:", min(value(m.CONG_ABS_T[t]) for t in m.T),
                         max(value(m.CONG_ABS_T[t]) for t in m.T))
    
    # PV curtailment tariff case
    pv_curt_tar = sum(value(m.PVcurt[i,t]) * DELTA_T for i in m.I for t in m.T)
    
    # Unmet charging demand tariff case
    unmet_energy_tar = sum(value(m.s_short[d]) for d in m.D)
    
    # Congestion tariff case
    cong_tar_total = sum(value(m.CONG_ABS_T[t]) for t in m.T)
    
    # Tariff objective value
    OBJ_tar = value(m.OBJ)
    # Tariff charging profile
    charging_tar_array = np.array([value(m.Pfleet[t]) for t in m.T])
    charging_shift = np.sum(np.abs(charging_tar_array - charging_base_array)) * DELTA_T
    

    total_capacity = sum(m_base.Emax_D[d] for d in m_base.D)
    soc_baseline = []
    soc_tariff = []
    for t in m_base.T:
        E_baseline = sum(value(m_base.Ed[d, t]) for d in m_base.D)
        E_tariff   = sum(value(m.Ed[d, t]) for d in m.D)
        soc_baseline.append(E_baseline / max(1e-9, total_capacity))
        soc_tariff.append(E_tariff / max(1e-9, total_capacity))
    soc_baseline = np.array(soc_baseline)
    soc_tariff   = np.array(soc_tariff)
    soc_earliest = soc_baseline.copy()
    soc_latest   = soc_baseline.copy()
    
    # =====================================================
    # VALIDATION — TARIFF CASE
    #  =====================================================
    # --------------------------
    # 0) Basic tariff sanity checks
    # --------------------------
    print("\n=== VALIDATION: TARIFF ===")

    # Power balance residual (network assumed enabled)
    max_balance_res_tar = max(abs(value(m.P_BAL[i, t].body())) for i in m.I for t in m.T)
    print("Max power balance residual (tariff):", float(max_balance_res_tar))

    # λ should be 0 when congestion index is ~0 (for your ladder logic)
    inconsistent_hours = [int(t) for t in m.T if value(m.lambda_tar[t]) > 1e-6 and value(m.CONG_INC_T[t]) < 1e-6]
    print("Hours with λ>0 but no congestion:", inconsistent_hours)


    # 1) Unmet charging demand (slack)

    unmet_energy_total = float(sum(value(m.s_short[d]) for d in m.D))  # pu-energy
    unmet_soc_avg = float(np.mean([
        value(m.s_short[d]) / max(1e-9, m.Emax_D[d])
        for d in m.D
    ]))


    # 2) Cost / energy metrics

    fleet_energy_cost = float(sum(value(m.price[t]) * value(m.Pfleet[t]) * DELTA_T for t in m.T))
    fleet_duos_payment = float(sum(value(m.lambda_tar[t]) * value(m.Pfleet[t]) * DELTA_T for t in m.T))
    fleet_net_payment = fleet_energy_cost + fleet_duos_payment
    
    ens_total = float(sum(max(0.0, value(m.ENS[i, t])) * DELTA_T for i in m.I for t in m.T))
    pv_curt   = float(sum(value(m.PVcurt[i, t]) * DELTA_T for i in m.I for t in m.T))
    delta_pvcurt = float(pv_curt - pv_curt_base)


    # 3) Congestion metrics (same index as ladder uses)

    cong_tar_abs_t = np.array([value(m.CONG_ABS_T[t]) for t in m.T], dtype=float)
    cong_tar_total = float(np.sum(cong_tar_abs_t))
    cong_inc_tar_total = sum(max(0.0, value(m.CONG_INC_T[t])) for t in m.T)
    peak_tar = float(np.max(cong_tar_abs_t)) if cong_tar_abs_t.size else 0.0
    
    sigma_tar_t = np.array([sum(value(m.sigma[l, t]) for l in m.L) for t in m.T], dtype=float)
    xi_tar_t    = np.array([sum(value(m.xi_low[i, t]) + value(m.xi_high[i, t]) for i in m.I) for t in m.T], dtype=float)
    
    sigma_abs_tar = float(np.sum(sigma_tar_t))
    xi_abs_tar    = float(np.sum(xi_tar_t))
    sigma_abs_base = float(np.sum(sigma_base_t))
    xi_abs_base    = float(np.sum(xi_base_t))
    

    # Baseline congestion totals (ensure consistent naming downstream)
    cong_base_abs_t = np.array([value(m_base.CONG_ABS_T[t]) for t in m_base.T], dtype=float)
    cong_base_total = float(np.sum(cong_base_abs_t))
    peak_base = float(np.max(cong_base_abs_t)) if cong_base_abs_t.size else 0.0
    delta_cong = cong_base_total - cong_tar_total

    print("\n--- Solver status ---")
    print("Baseline status:", results_base.solver.status)
    print("Baseline termination:", results_base.solver.termination_condition)
    print("Tariff status:", results_tar.solver.status)
    print("Tariff termination:", results_tar.solver.termination_condition)

    print("\n--- Tariff λ values (unique) ---")
    print(sorted(set(round(value(m.lambda_tar[t]), 6) for t in m.T)))
    
    print("\n=== Congestion Reduction Test ===")
    # PV curtailment comparison
    pv_curt_base = sum(
        value(m_base.PVcurt[i, t])
        for i in m_base.I
        for t in m_base.T
    )
    pv_curt_tar = sum(
        value(m.PVcurt[i, t])
        for i in m.I
        for t in m.T
    )
    pv_curt_base_t = np.array([
        sum(value(m_base.PVcurt[i, t]) for i in m_base.I)
        for t in m_base.T
    ])
    pv_curt_tar_t = np.array([
        sum(value(m.PVcurt[i, t]) for i in m.I)
        for t in m.T
    ])
    print("\n=== PV Curtailment Comparison ===")
    print(f"Baseline PV curtailment : {pv_curt_base:.6f}")
    print(f"Tariff PV curtailment   : {pv_curt_tar:.6f}")
    print(f"Reduction               : {pv_curt_base - pv_curt_tar:.6f}")

    print(f"Baseline total congestion : {cong_base_total:.6f}")
    print(f"Tariff total congestion   : {cong_tar_total:.6f}")
    print(f"Reduction                 : {cong_base_total - cong_tar_total:.6f}")


    # 4) Charging shift

    charging_tariff_array = np.array([value(m.Pfleet[t]) for t in m.T], dtype=float)
    charging_shift = float(np.sum(np.abs(charging_tariff_array - charging_base_array)) * DELTA_T)

    delta_cong = cong_base_total - cong_tar_total


    # 5) Fleet cost (objective-consistent payment proxy)

    fleet_cost = float(sum( (value(m.price[t]) + value(m.lambda_tar[t])) * value(m.Pfleet[t]) * DELTA_T for t in m.T))
    thermal_base = [ sum(value(m_base.sigma[l,t]) for l in m_base.L) for t in m_base.T ]
    voltage_base = [sum(value(m_base.xi_low[i,t]) + value(m_base.xi_high[i,t]) for i in m_base.I) for t in m_base.T]


    # 6) Hourly series for plotting / storyline
    hourly = {
        "pv": np.array([sum(PV_dict[(i, t)] for i in I) for t in m.T], dtype=float),
        "charging_caseF": charging_caseF_array,
        "pv_caseF": pv_caseF_array,
        "load_caseF": load_caseF_array,
        "thermal_caseF": thermal_caseF_array,
        "voltage_caseF": voltage_caseF_array,
        "pv_curt_tar": np.array([sum(value(m.PVcurt[i, t]) for i in m.I) for t in m.T], dtype=float),
        "pv_curt_base": np.array([sum(value(m_base.PVcurt[i, t]) for i in m_base.I) for t in m_base.T], dtype=float),
        "ens_tar": np.array([sum(value(m.ENS[i, t]) for i in m.I) for t in m.T], dtype=float),
        "ens_base": np.array([sum(value(m_base.ENS[i, t]) for i in m_base.I) for t in m_base.T], dtype=float),
        "charging_tariff": charging_tariff_array,
        "charging_base": np.array([value(m_base.Pfleet[t]) for t in m_base.T], dtype=float),
        "thermal_pu": sigma_tar_t,
        "thermal_base_pu": np.array(sigma_base_t, dtype=float),
        "voltage_pu2": xi_tar_t,
        "voltage_base_pu2": np.array(xi_base_t, dtype=float),
        "lambda": np.array([value(m.lambda_tar[t]) for t in m.T], dtype=float),
    }

    print(f"\nFleet scale = {fleet_scale}")


    if np.std(cong_tar_abs_t) > 1e-9 and np.std(hourly["lambda"]) > 1e-9:
        corr = np.corrcoef(cong_tar_abs_t, hourly["lambda"])[0,1]
    else:
        corr = 0.0
    print("Correlation congestion–λ:", float(corr))




         
    # 7) Payment breakdown (clean, no undefined external names)
    def payment_breakdown(m_base, m, DELTA_T, cong0_total, cong_base_total, cong_tar_total, peak_base, peak_tar, hourly): 
        Tlist = list(m.T)
        P_base = np.array([value(m_base.Pfleet[t]) for t in Tlist], dtype=float)
        P_tar  = np.array([value(m.Pfleet[t])      for t in Tlist], dtype=float)
        price  = np.array([value(m.price[t])       for t in Tlist], dtype=float)
        lam    = np.array([value(m.lambda_tar[t])  for t in Tlist], dtype=float)
        
        base_energy = float(np.sum(price * P_base * DELTA_T))
        tar_energy  = float(np.sum(price * P_tar  * DELTA_T))
        
        tar_duos = float(np.sum(lam * P_tar * DELTA_T))
        base_duos_if_tar = float(np.sum(lam * P_base * DELTA_T))
        
        base_total = base_energy
        tar_total  = tar_energy + tar_duos
        
        schedule_effect_energy = tar_energy - base_energy
        duos_effect = tar_duos
        alignment = tar_duos - base_duos_if_tar  # <0 means avoided λ hours vs baseline
        
        return {
            "base_energy": base_energy,
            "tar_energy": tar_energy,
            "tar_duos": tar_duos,
            "base_duos_if_tar": base_duos_if_tar,
            "base_total": base_total,
            "tar_total": tar_total,
            
            "congestion_no_fleet": float(cong0_total),
            "congestion_abs_base": float(cong_base_total),
            "congestion_abs_tar": float(cong_tar_total),
            "delta_cong": delta_cong,
        
            "avg_lambda": float(np.mean(hourly["lambda"])),
            "peak_lambda": float(np.max(hourly["lambda"])),
            
            "schedule_effect_energy": float(schedule_effect_energy),
            "duos_effect": float(duos_effect),
            "alignment_vs_baseline": float(alignment),
            
            "peak_base": float(peak_base),
            "peak_tar": float(peak_tar),
            "peak_improvement": float(peak_base - peak_tar),
            
            "series": {
                "P_base": P_base, "P_tar": P_tar, "price": price, "lambda": lam,
                "hourly_energy_base": price * P_base * DELTA_T,
                "hourly_energy_tar":  price * P_tar  * DELTA_T,
                "hourly_duos_tar":    lam   * P_tar  * DELTA_T,
                "hourly_total_base":  price * P_base * DELTA_T,
                "hourly_total_tar":  (price + lam) * P_tar * DELTA_T,
                "hourly_delta_total": (price + lam) * P_tar * DELTA_T - price * P_base * DELTA_T,
            }
        }
    
    pay = payment_breakdown(
        m_base, m, DELTA_T,
        cong0_total=cong0_total,
        cong_base_total=cong_base_total,
        cong_tar_total=cong_tar_total,
        peak_base=peak_base,
        peak_tar=peak_tar,
        hourly=hourly
    )
    
    print("\n=== Fleet payment breakdown (€/day) ===")
    # Fleet charging profile (hourly)
    charging_base_array = np.array([
        sum(value(m_base.Pd[d, t]) for d in m_base.D)
        for t in m_base.T
    ])
    charging_tar_array = np.array([
        sum(value(m.Pd[d, t]) for d in m.D)
        for t in m.T
    ])
    print("\n=== Fleet charging profile (MW) ===")
    for t in m.T:
        print(
            f"t={t:02d}  "
            f"P_base={charging_base_array[t]:.3f}  "
            f"P_tar={charging_tar_array[t]:.3f}"
        )

    print(f"Base energy-only      : {pay['base_energy']:.2f}")
    print(f"Tariff energy-only    : {pay['tar_energy']:.2f}")
    print(f"Tariff DUoS add-on    : {pay['tar_duos']:.2f}")
    print(f"TOTAL base            : {pay['base_total']:.2f}")
    print(f"TOTAL tariff          : {pay['tar_total']:.2f}")
    print(f"Δ due to schedule (price*P) : {pay['schedule_effect_energy']:+.2f}")
    print(f"Δ due to DUoS (λ*P)         : {pay['duos_effect']:+.2f}")
    print(f"Tariff alignment vs baseline (λ*P_tar - λ*P_base): {pay['alignment_vs_baseline']:+.2f}")
    
    d = pay["series"]["hourly_delta_total"]
    idx = np.argsort(d)
    print("\n=== Biggest savings hours (top 5) ===")
    for j in idx[:5]:
        print(
            f"t={j:02d}  Δ€={d[j]:+.2f}  "
            f"P_base={pay['series']['P_base'][j]:.3f}  "
            f"P_tar={pay['series']['P_tar'][j]:.3f}  "
            f"price={pay['series']['price'][j]:.1f}  "
            f"λ={pay['series']['lambda'][j]:.1f}"
        )
        
    print("\n=== Biggest cost increase hours (top 5) ===")
    for j in idx[-5:][::-1]:
        print(
            f"t={j:02d}  Δ€={d[j]:+.2f}  "
            f"P_base={pay['series']['P_base'][j]:.3f}  "
            f"P_tar={pay['series']['P_tar'][j]:.3f}  "
            f"price={pay['series']['price'][j]:.1f}  "f"λ={pay['series']['lambda'][j]:.1f}"
        )
        
          
    obj_base = next(m_base.component_data_objects(pyo.Objective))
    obj_tar  = next(m.component_data_objects(pyo.Objective))
    
    out = {
        "ENS": ens_total,
        "pv_curt": pv_curt,                 # tariff case total PVcurt energy
        "fleet_cost": fleet_cost,
        "unmet_energy_total": unmet_energy_total,
        "unmet_soc_avg": unmet_soc_avg,
        "base_load_only": base_load_only,

        "soc_baseline": soc_baseline,
        "soc_tariff": soc_tariff,
        "soc_earliest": soc_earliest,
        "soc_latest": soc_latest,
        
        # --- BASELINE / TARIFF PV curtailment totals (the keys you crash on) ---
        "pv_curt_tar":  pv_curt_tar,
        "pv_curt_base": pv_curt_base,
        # --- unmet charging baseline/tariff ---
        "unmet_energy_base": unmet_energy_base,
        "unmet_energy_tar":  unmet_energy_tar,
        # --- congestion totals ---
        "cong_base_total": cong_base_total,
        "cong_tar_total":  cong_tar_total,
        "delta_cong":      delta_cong,
        
        # --- objective values ---
        "OBJ_base": OBJ_base,
        "OBJ_tar":  OBJ_tar,

        "fleet_energy": energy_MWh,
        "avg_lambda": float(np.mean(hourly["lambda"])),
        "peak_lambda": float(np.max(hourly["lambda"])),
        
        # --- flexibility metric used in plots ---
        "charging_shift": charging_shift,
        
        # --- store hourly series for storyline plots ---
        "hourly": hourly,
    }
    return out

# ==========================================================
# EXPERIMENT — C_SHORT sensitivity
# ==========================================================
C_SHORT_values = [100,300,500, 600, 700, 900, 1000, 1400]
cshort_table= []
# ======================================================================
# SECTION 3 — SOLVER
# ======================================================================


#fleet_sizes = [len(B), 2*len(B), 3*len(B), 4*len(B)]
fleet_sizes = [1,2,3,4,5,6,7]  # 1, 3, 5 

results = {}

for C_SHORT_val in C_SHORT_values:

    # update global parameter
    C_SHORT = C_SHORT_val

    print(f"Running experiments with C_SHORT = {C_SHORT}")

    for k in fleet_sizes:
        print(f"Solving for fleet size = {k}")
        try:
            res = solve_for_fleet_size(k)
            if res is None:
                continue
            # Extract values (tariff case only)
            soc_final = res["soc_tariff"][-1]      # final fleet SOC
            pv_curt_final = res["pv_curt_tar"]     # total PV curtailment
            cong_resolved = res["delta_cong"]      # congestion reduction

            cshort_table.append({
                "C_SHORT": C_SHORT_val,
                "fleet_size": k,
                "SOC_final": soc_final,
                "PV_curtailment": pv_curt_final,
                "Congestion_resolved": cong_resolved
            })
            if res is None:
                print(f"Fleet size {k}: infeasible / no result stored.")
                continue
            
            results.setdefault(C_SHORT_val, {})[k] = res
            print("  ENS:", res["ENS"])
            print("  PV Curt    :", res["pv_curt"])
            print("  Unmet energy (pu-E):", res["unmet_energy_total"])
            print("  Unmet SOC avg      :", res["unmet_soc_avg"])
            print("  Fleet Cost :", res["fleet_cost"])

            summary_rows.append({
                "C_SHORT": C_SHORT_val,
                "fleet_size": k,
                "ENS": res["ENS"],
                "PV_curtailment": res["pv_curt"],
                "Fleet_cost": res["fleet_cost"],
                "Congestion_base": res["cong_base_total"],
                "Congestion_tariff": res["cong_tar_total"],
                "Congestion_reduction": res["delta_cong"],
                "Charging_shift": res["charging_shift"]
            })
            hourly = res["hourly"]
            for t in range(len(hourly["lambda"])):
                hourly_rows.append({
                    "C_SHORT": C_SHORT_val,
                    "charging_caseF": hourly["charging_caseF"][t],
                    "fleet_size": k,
                    "hour": t,
                    "lambda": hourly["lambda"][t],
                    "charging_base": hourly["charging_base"][t],
                    "charging_tariff": hourly["charging_tariff"][t],
                    "thermal_base": hourly["thermal_base_pu"][t],
                    "thermal_tariff": hourly["thermal_pu"][t],
                    "voltage_base": hourly["voltage_base_pu2"][t],
                    "voltage_tariff": hourly["voltage_pu2"][t],
                    "pv_curt_base": hourly["pv_curt_base"][t],
                    "pv_curt_tar": hourly["pv_curt_tar"][t],
                    "ens_base": hourly["ens_base"][t],
                    "ens_tar": hourly["ens_tar"][t]
                })
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            continue



print("Min Smax:", min(Smax_map.values()))
print("Max Smax:", max(Smax_map.values()))




valid_keys = [kk for kk,vv in results.items() if vv is not None]
if not valid_keys:
    print("No successful runs.")
else:
    last_cshort = max(valid_keys)
    if last_cshort in results and len(results[last_cshort]) > 0:
        last_fleet = max(results[last_cshort].keys())
    last_result = results[last_cshort][last_fleet]
    print("C_SHORT:", last_cshort)
    print("Fleet size:", last_fleet)
    print("  ENS        :", last_result["ENS"])
    print("  PV Curt    :", last_result["pv_curt"])
    print("  Fleet Cost :", last_result["fleet_cost"])


summary_df = pd.DataFrame(summary_rows)
hourly_df = pd.DataFrame(hourly_rows)
cshort_df = pd.DataFrame(cshort_table)

print("\n================ C_SHORT SENSITIVITY TABLE ================")
print(cshort_df)

# save to Excel
with pd.ExcelWriter("C:\\Users\\HP\\Desktop\\Code\\simulation_results.xlsx" , engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary_results", index=False)
    hourly_df.to_excel(writer, sheet_name="hourly_results", index=False)
    cshort_df.to_excel(writer, sheet_name="C_SHORT_sensitivity", index=False)
# ==========================================================
# PLOTTING
# ==========================================================
# COLORS

COL_BASE = "#FFD84D"   # lemon yellow
COL_TAR   = "#2E86C1"   # strong blue → Tariff
COL_CASEF = "#3FA34D"   # muted green → Case F

COL_PV_GEN  = "#E3B505"  # warm yellow → PV generation
COL_PV_USED = "#F2D16B"  # soft yellow
COL_PV_CURT = "#D98C3A"  # amber/orange → curtailed PV

COL_CONG  = "#C0392B"   # deep red → congestion
COL_SHIFT = "#D35454"   # muted red → charging shift

GRID_COL = "#C8C8C8"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12
})

if len(results) == 0:
    print("No successful runs — skipping plots.")
    import sys
    sys.exit(0)


selected_cshort = C_SHORT_values[-1]
fleet_sizes = sorted(results[selected_cshort].keys())

# keep only fleet sizes that produced a valid result dictionary
valid_sizes = [n for n in fleet_sizes if results[selected_cshort].get(n) is not None]

if len(valid_sizes) == 0:
    print("No valid result dictionaries available for plotting.")
    import sys
    sys.exit(0)

# Converting fleet scale into installed fleet charging capacity (MW)
fleet_capacity_MW = np.array(valid_sizes) * Pmax_fleet * S_BASE_MVA



def plot_soc_flexibility(
        soc_baseline,
        soc_tariff,
        soc_fastest,
        soc_latest,
        min_departure_soc=0.8,
        fleet_label="Fleet size 3"
    ):

    soc_baseline = np.array(soc_baseline)
    soc_tariff = np.array(soc_tariff)
    soc_fastest = np.array(soc_fastest)
    soc_latest = np.array(soc_latest)

    nT = len(soc_baseline)
    hours = np.arange(1, nT + 1)

    plt.figure(figsize=(12,6))

    # flexibility envelope
    plt.fill_between(
        hours,
        soc_latest,
        soc_fastest,
        color="#4C78A8",
        alpha=0.35,
        label="Flexibility envelope"
    )

    # tariff SOC (BLUE)
    plt.plot(
        hours,
        soc_tariff,
        color="#1f77b4",
        linewidth=2.5,
        label="Tariff SOC"
    )

    # baseline SOC (YELLOW)
    plt.plot(
        hours,
        soc_baseline,
        color="#FFD84D",
        linewidth=2.5,
        label="Baseline SOC"
    )

    # minimum departure SOC
    plt.axhline(
        min_departure_soc,
        linestyle=":",
        color="black",
        linewidth=2,
        label="Min departure SOC"
    )

    plt.xlabel("Hour")
    plt.ylabel("State of Charge")
    plt.title(f"Fleet SOC flexibility and service guarantee — {fleet_label}")

    plt.xticks(np.arange(1, nT+1, 1))
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------
# Aggregated arrays across fleet sizes
# ----------------------------------------------------------

congestion_severity = np.array([
    results[selected_cshort][n].get("cong_tar_total", 0.0) for n in valid_sizes
], dtype=float)

pv_curtailment = np.array([
    results[selected_cshort][n].get("pv_curt", 0.0) for n in valid_sizes
], dtype=float)

ens_total = np.array([
    results[selected_cshort][n].get("ENS", 0.0) for n in valid_sizes
], dtype=float)

fleet_cost_loss = np.array([
    results[selected_cshort][n].get("fleet_cost", 0.0) for n in valid_sizes
], dtype=float)

pv_base = np.array([
    results[selected_cshort][n].get("pv_curt_base", 0.0) for n in valid_sizes
], dtype=float)

pv_tar = np.array([
    results[selected_cshort][n].get("pv_curt_tar", 0.0) for n in valid_sizes
], dtype=float)

unmet_base = np.array([
    results[selected_cshort][n].get("unmet_energy_base", 0.0) for n in valid_sizes
], dtype=float)

unmet_tar = np.array([
    results[selected_cshort][n].get("unmet_energy_tar", 0.0) for n in valid_sizes
], dtype=float)

cong_improvement = np.array([
    results[selected_cshort][n].get("delta_cong", 0.0) for n in valid_sizes
], dtype=float)

flexibility = np.array([
    results[selected_cshort][n].get("charging_shift", 0.0) for n in valid_sizes
], dtype=float)

cost_base = np.array([
    results[selected_cshort][n].get("OBJ_base", 0.0) for n in valid_sizes
], dtype=float)

cost_tar = np.array([
    results[selected_cshort][n].get("OBJ_tar", 0.0) for n in valid_sizes
], dtype=float)


# pu·h → MWh
pv_base = pv_base * S_BASE_MVA 
pv_tar = pv_tar * S_BASE_MVA 
unmet_base = unmet_base * S_BASE_MVA
unmet_tar = unmet_tar * S_BASE_MVA
flexibility = flexibility * S_BASE_MVA
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes = axes.flatten()

axes[0].plot(fleet_capacity_MW, pv_base, marker="o", color=COL_BASE, label="Baseline")
axes[0].plot(fleet_capacity_MW, pv_tar, marker="o", color=COL_TAR, label="Tariff")
axes[0].set_xlabel("Installed fleet charging capacity (MW)")
axes[0].set_ylabel("PV curtailment (MWh)")
axes[0].set_title("PV curtailment")
axes[0].grid(True, alpha=0.3, color=GRID_COL)
axes[0].legend(loc="upper left", frameon=False)

axes[1].plot(fleet_capacity_MW, unmet_base, marker="o", color=COL_BASE, label="Baseline")
axes[1].plot(fleet_capacity_MW, unmet_tar, marker="o", color=COL_TAR, label="Tariff")
axes[1].set_xlabel("Installed fleet charging capacity (MW)")
axes[1].set_ylabel("Unmet charging demand (MWh)")
axes[1].set_title("Unmet charging demand")
axes[1].grid(True, alpha=0.3, color=GRID_COL)
axes[1].legend(loc="upper left", frameon=False)

axes[2].plot(fleet_capacity_MW, cong_improvement, marker="o", color=COL_CONG)
axes[2].set_xlabel("Installed fleet charging capacity (MW)")
axes[2].set_ylabel("Congestion reduction")
axes[2].set_title("Tariff-induced congestion reduction")
axes[2].grid(True, alpha=0.3, color=GRID_COL)

axes[3].plot(fleet_capacity_MW, flexibility, marker="o", color=COL_SHIFT)
axes[3].set_xlabel("Installed fleet charging capacity (MW)")
axes[3].set_ylabel("Tariff-induced charging shift (MWh)")
axes[3].set_title("Tariff-induced charging shift")
axes[3].grid(True, alpha=0.3, color=GRID_COL)

# Apply ticks globally
for ax in axes:
    ax.set_xticks(fleet_capacity_MW)

plt.tight_layout()
plt.show()


if selected_cshort in results and len(results[selected_cshort]) > 0:
    last_valid_fleet = max(results[selected_cshort].keys())
    #plot_tariff_vs_congestion(results[selected_cshort][last_valid_fleet], last_valid_fleet)
else:
    print("No valid fleet results available for plotting.")

#SYSTEM IMPACT OF THE TARIFF
def plot_system_impact(result, fleet_size):

    data = result["hourly"]
    cong_base = data["thermal_base_pu"] + data["voltage_base_pu2"]
    cong_tar  = data["thermal_pu"] + data["voltage_pu2"]

    pv_base = data["pv_curt_base"]
    pv_tar  = data["pv_curt_tar"] 

    Tplot = np.arange(len(cong_base))

    fig, axes = plt.subplots(2,1,figsize=(10,8),sharex=True)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].step(Tplot, cong_base, where="post", color=COL_BASE, label="Baseline")
    axes[0].step(Tplot, cong_tar, where="post", color=COL_TAR, label="Tariff")
    axes[0].set_ylabel("Congestion index")
    axes[0].set_title(f"Network congestion (fleet size {fleet_size})")
    axes[0].legend(loc="upper left", frameon=False)
    
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    axes[1].step(Tplot, pv_base, where="post", color=COL_BASE, label="Baseline")
    axes[1].step(Tplot, pv_tar, where="post", color=COL_TAR, label="Tariff")
    axes[1].set_ylabel("PV curtailment (MW)")
    axes[1].set_xlabel("Hour")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)


    plt.tight_layout()
    plt.show()

# -----------------------------------------
# Compute fleet availability profile
# -----------------------------------------
def fleet_availability(truck_data, nT=24):

    availability = np.zeros(nT)
    

    for _, row in truck_data.iterrows():

        t_arr = int(row["t_arr"])
        t_dep = int(row["t_dep"])

        if t_dep > t_arr:
            availability[t_arr:t_dep] += 1
        else:
            # overnight stay
            availability[t_arr:] += 1
            availability[:t_dep] += 1

    return availability

# ----------------------------------------------------------
# Storyline plot for one fleet size
# ----------------------------------------------------------
def plot_storyline(result, fleet_size):
    if result is None:
        return

    data = result.get("hourly", {})
    base_load_only = np.array(result.get("base_load_only", []), dtype=float)
    if not data:
        print(f"Skipping plot for fleet size {fleet_size}: no hourly data found.")
        return


    # Read hourly arrays safely
    pv_profile = np.array(data.get("pv", []), dtype=float)
    pv_base_hourly = pv_profile - np.array(data.get("pv_curt_base", []), dtype=float)
    pv_tar_hourly  = pv_profile - np.array(data.get("pv_curt_tar", []), dtype=float)
    
    pv_caseF = np.array(data.get("pv_caseF", []), dtype=float)
    load_caseF = np.array(data.get("load_caseF", []), dtype=float)
    
    thermal_caseF = np.array(data.get("thermal_caseF", []), dtype=float)
    voltage_caseF = np.array(data.get("voltage_caseF", []), dtype=float)

    thermal_tar = np.array(data.get("thermal_pu", []), dtype=float)
    voltage_tar = np.array(data.get("voltage_pu2", []), dtype=float)

    thermal_base = np.array(data.get("thermal_base_pu", []), dtype=float)
    voltage_base = np.array(data.get("voltage_base_pu2", []), dtype=float)

    lam = np.array(data.get("lambda", []), dtype=float) / 10.0

    charge_base = np.array(data.get("charging_base", []), dtype=float)
    charge_tar  = np.array(data.get("charging_tariff", []), dtype=float)
    charge_caseF = np.array(data.get("charging_caseF", []), dtype=float)


    ens_base_hourly = np.array(data.get("ens_base", []), dtype=float)
    ens_tar_hourly  = np.array(data.get("ens_tar", []), dtype=float)

    # Determine common length safely
    lengths = [
        len(base_load_only),
        len(pv_profile), len(pv_caseF), len(load_caseF),
        len(pv_profile), len(pv_caseF), len(load_caseF),
        len(pv_base_hourly), len(pv_tar_hourly),
        len(thermal_caseF), len(voltage_caseF),
        len(thermal_tar), len(voltage_tar),
        len(thermal_base), len(voltage_base),
        len(lam), len(charge_base), len(charge_tar),
        len(ens_base_hourly), len(ens_tar_hourly)
    ]
    lengths = [L for L in lengths if L > 0]

    if len(lengths) == 0:
        print(f"Skipping plot for fleet size {fleet_size}: all hourly arrays are empty.")
        return

    nT = min(lengths)
    MW = S_BASE_MVA
    MWh = S_BASE_MVA
    Tplot = np.arange(nT)

    availability = fleet_availability(bev, nT)

    max_av = max(1e-9, availability.max())
    availability_norm = availability / max_av

    pv_profile = pv_profile[:nT] if len(pv_profile) else np.zeros(nT)
    pv_caseF = pv_caseF[:nT] if len(pv_caseF) else np.zeros(nT)
    load_caseF = load_caseF[:nT] if len(load_caseF) else np.zeros(nT)
    
    thermal_caseF = thermal_caseF[:nT] if len(thermal_caseF) else np.zeros(nT)
    voltage_caseF = voltage_caseF[:nT] if len(voltage_caseF) else np.zeros(nT)

    # Trim everything to same length
    base_load_only = base_load_only[:nT] if len(base_load_only) else np.zeros(nT)
    pv_base_hourly = pv_base_hourly[:nT] if len(pv_base_hourly) else np.zeros(nT)
    pv_tar_hourly  = pv_tar_hourly[:nT] if len(pv_tar_hourly) else np.zeros(nT)

    thermal_tar = thermal_tar[:nT] if len(thermal_tar) else np.zeros(nT)
    voltage_tar = voltage_tar[:nT] if len(voltage_tar) else np.zeros(nT)

    thermal_base = thermal_base[:nT] if len(thermal_base) else np.zeros(nT)
    voltage_base = voltage_base[:nT] if len(voltage_base) else np.zeros(nT)

    lam = lam[:nT] if len(lam) else np.zeros(nT)

    charge_base = charge_base[:nT] if len(charge_base) else np.zeros(nT)
    charge_tar  = charge_tar[:nT] if len(charge_tar) else np.zeros(nT)
    charge_caseF = charge_caseF[:nT] if len(charge_caseF) else np.zeros(nT)

    ens_base_hourly = ens_base_hourly[:nT] if len(ens_base_hourly) else np.zeros(nT)
    ens_tar_hourly  = ens_tar_hourly[:nT] if len(ens_tar_hourly) else np.zeros(nT)
    
    # Convert to plotting units
    pv_gen = pv_profile * MW
    pv_base = pv_base_hourly * MW
    pv_tar = pv_tar_hourly * MW
    
    load_caseF_total = (base_load_only + charge_caseF) * MW
    load_base_total  = (base_load_only + charge_base) * MW
    load_tar_total   = (base_load_only + charge_tar) * MW
    
    cong_caseF = thermal_caseF + voltage_caseF
    cong_base  = thermal_base + voltage_base
    cong_tar   = thermal_tar + voltage_tar
    
    charge_caseF_MW = charge_caseF * MW
    charge_base_MW  = charge_base * MW
    charge_tar_MW   = charge_tar * MW
    delta_base = charge_base_MW - charge_caseF_MW
    delta_tar  = charge_tar_MW - charge_caseF_MW
    Tplot = np.arange(1, nT + 1)
   

    # ============================================================
    # FIGURE 1: GRID-LEVEL IMPACT
    # ============================================================

    fig1, axes = plt.subplots(
        3, 1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]}
        )
    fig1.suptitle(
        f"Grid-level impact of congestion-dependent DUoS tariff, fleet size = {fleet_size}",
        fontsize=14,
        y=0.98
    )

    # 1) Total load
    axes[0].step(
        Tplot, load_caseF_total,
        where="post",
        color=COL_CASEF,
        linestyle=":",
        linewidth=2,
        label="Case1"
        )
    
    axes[0].step(
        Tplot, load_base_total,
        where="post",
        color=COL_BASE,
        linestyle="--",
        linewidth=2,
        label="Case 2"
    )
    
    axes[0].step(
        Tplot, load_tar_total,
        where="post",
        color=COL_TAR,
        linewidth=2,
        label="Case 3"
    )
    axes[0].set_ylabel("Total load (MW)")
    axes[0].set_title("Total feeder load")
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    
    
    # 2) Congestion
    axes[1].step(
        Tplot, cong_caseF[:nT],
        where="post",
        color=COL_CASEF,
        linestyle=":",
        linewidth=2,
        label="Case 1"
        )
    
    axes[1].step(
        Tplot, cong_base[:nT],
        where="post",
        color=COL_BASE,
        linestyle="--",
        linewidth=2,
        label="Case 2"
        
    )
    
    axes[1].step(
        Tplot, cong_tar[:nT],
        where="post",
        color=COL_TAR,
        linewidth=2,
        label="Case 3"
    )
    
    axes[1].set_ylabel("Congestion index")
    axes[1].set_title("Congestion index")
    axes[1].legend(loc="upper left", frameon=False, ncol=3)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # 3) DUoS tariff signal
    
    axes[2].step(
        Tplot, lam,
        where="post",
        linewidth=2.5,
        color=COL_TAR,
        label=r"DUoS tariff $\lambda$"
    )
    
    axes[2].set_ylabel(r"$\lambda$ (€/MWh)")
    axes[2].set_xlabel("Hour")
    axes[2].set_title("DUoS tariff signal")
    axes[2].legend(loc="upper left", frameon=False)
    axes[2].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axes[2].set_yticks([0, 20, 40, 60, 80, 100, 140])
    
    axes[-1].set_xticks(np.arange(1, nT + 1, 1))
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.45)
    plt.show()


    # ============================================================
    # FIGURE 2: FLEET-RESPONSE IMPACT
    # # ============================================================

    fig2, axes = plt.subplots(
        3, 1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.3, 1.0]}
    )
    
    fig2.suptitle(
        f"Fleet response to congestion-dependent DUoS tariff, fleet size = {fleet_size}",
        fontsize=14,
        y=0.98
        )
    
    # 1) PV generation and utilization
    axes[0].step(
        Tplot, pv_gen,
        where="post",
        color=COL_PV_GEN,
        linewidth=2.5,
        label="PV generation"
    )
    
    axes[0].step(
        Tplot, pv_base,
        where="post",
        color=COL_BASE,
        linestyle="--",
        linewidth=2,
        label="PV used (baseline)"
    )
    
    axes[0].step(
        Tplot, pv_tar,
        where="post",
        color=COL_TAR,
        linewidth=2,
        label="PV used (tariff)"
    )
    axes[0].set_ylabel("PV power (kW)")
    axes[0].set_title("PV generation and utilization")
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    
    # 2) Fleet charging schedule
    # 
    axes[1].step(
        Tplot, charge_caseF_MW,
        where="post",
        color=COL_CASEF,
        linestyle=":",
        linewidth=2,
        label="Case 1"
    )
    
    axes[1].step(
        Tplot, charge_base_MW,
        where="post",
        color=COL_BASE,
        linestyle="--",
        linewidth=2,
        label="Case 2"
    )
    
    axes[1].step(
        Tplot, charge_tar_MW,
        where="post",
        color=COL_TAR,
        linewidth=2,
        label="Case 3"
    )
    
    axes[1].fill_between(
        Tplot,
        0,
        availability_norm * max(charge_tar_MW),
        color="lightgrey",
        alpha=0.3,
        label="Fleet available"
    )
    
    axes[1].set_ylabel("Fleet charging (MW)")
    axes[1].set_title("Fleet charging schedule and availability")
    axes[1].legend(loc="upper left", frameon=False, ncol=4)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    
    # 3) Charging shif
    width = 0.4
    axes[2].bar(
        Tplot - width / 2,
        delta_base,
        width=width,
        color=COL_BASE,
        label="Case 2"
    )
    axes[2].bar(
        Tplot + width / 2,
        delta_tar,
        width=width,
        color=COL_TAR,
        label="Case 3"
    )
    
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("Charging shift (MW)")
    axes[2].set_xlabel("Hour")
    axes[2].set_title("Charging shift relative to uncontrolled charging")
    axes[2].legend(loc="upper left", frameon=False, ncol=2)
    axes[2].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    
    axes[-1].set_xticks(np.arange(1, nT + 1, 1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.45)
    plt.show()


for k in valid_sizes:
    plot_storyline(results[selected_cshort][k], k)



# ==========================================================
# RQ2 — LAMBDA CURVES BY FLEET SIZE, FIXED CASE
# ==========================================================

def plot_rq2_lambda_fleet_size_fixed(
        results,
        selected_cshort,
        fleet_sizes_to_plot=None,
        title="RQ2: Effect of HD-BEV fleet size on congestion-activated grid tariff"
    ):

    if selected_cshort not in results:
        print("Selected C_SHORT not found in results.")
        return

    fleet_sizes = sorted(results[selected_cshort].keys())

    if fleet_sizes_to_plot is None:
        fleet_sizes_to_plot = fleet_sizes

    valid_sizes = [
        n for n in fleet_sizes_to_plot
        if n in results[selected_cshort]
        and results[selected_cshort][n] is not None
        and "hourly" in results[selected_cshort][n]
        and "lambda" in results[selected_cshort][n]["hourly"]
    ]

    if len(valid_sizes) == 0:
        print("No valid fleet-size lambda results available for plotting.")
        return

    Tplot = np.arange(1, 25)

    plt.figure(figsize=(11, 5.5))

    max_lambda = -np.inf
    max_hour = None
    max_fleet = None

    for n in valid_sizes:

        lam = np.array(
            results[selected_cshort][n]["hourly"]["lambda"],
            dtype=float
        ) / 10.0   # €/pu → €/MWh

        lam = lam[:24]

        plt.step(
            Tplot,
            lam,
            where="post",
            linewidth=2.2,
            marker="o",
            label=f"Fleet size x{n}"
        )

        local_idx = int(np.argmax(lam))
        if lam[local_idx] > max_lambda:
            max_lambda = lam[local_idx]
            max_hour = Tplot[local_idx]
            max_fleet = n

    # Highlight maximum activated tariff
    if max_hour is not None:
        plt.scatter(
            max_hour,
            max_lambda,
            s=90,
            color=COL_CONG,
            zorder=5,
            label="Maximum activated tariff"
        )

        plt.annotate(
            f"Max λ = {max_lambda:.0f} €/MWh\nFleet size x{max_fleet}, hour {max_hour}",
            xy=(max_hour, max_lambda),
            xytext=(max_hour + 1, min(140, max_lambda + 15)),
            arrowprops=dict(arrowstyle="->", linewidth=1.2),
            fontsize=10
        )

    plt.xlabel("Hour")
    plt.ylabel(r"DUoS tariff $\lambda$ (€/MWh)")
    plt.title(title)

    plt.xticks(np.arange(1, 25, 1))
    plt.yticks([0, 20, 40, 60, 80, 100, 140])
    plt.ylim(0, 145)

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=GRID_COL)
    plt.legend(loc="upper left", frameon=False, ncol=2)

    plt.tight_layout()
    plt.show()


# Run for fixed case
selected_cshort = C_SHORT_values[-1]

plot_rq2_lambda_fleet_size_fixed(
    results,
    selected_cshort,
    fleet_sizes_to_plot=[1, 2, 3, 4, 5]
)