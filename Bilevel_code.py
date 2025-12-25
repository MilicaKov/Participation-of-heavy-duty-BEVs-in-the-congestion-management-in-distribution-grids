import pyomo.environ as pyo
import pandas as pd
from collections import defaultdict
import numpy as np
from pyomo.environ import value


#Switches 
USE_KKT     = True   # bilevel via KKT (Option A: relaxed stationarity)
USE_NETWORK = True   # include LinDistFlow network


# FILE PATHS 
BEV_XLSX        = r"C:\Users\HP\source\repos\Function_Solution\Inputs\BEV_parameters.xlsx"
BASELINE_XLSX   = r"C:\Users\HP\source\repos\Function_Solution\Inputs\Baseline.xlsx"
TIMESERIES_XLSX = r"C:\Users\HP\source\repos\Function_Solution\Inputs\Time_Series.xlsx"


# CONSTANTS 
DELTA_T = 1.0  # hours

C_FLEET = 2000   # €/MWh

# Upper-level weights
C_VOLT   = 1e6     # voltage deviation slack penalty
C_THERM  = 1e6    # thermal slack penalty
C_ENS    = 1e7      # ENS penalty
C_PVCURT = 1e6     # PV curtailment penalty

C_KKT    = 1e4      # penalty on relaxed stationarity slacks
W_EPS = 1e8

# Tariff bounds/ramping
LAMBDA_MIN = 0
LAMBDA_MAX = 5000
DELTA_LAMBDA_MAX = 1e6

# Fleet value of energy at departure (€/MWh)
V_E = 5000  

# Congestion-to-tariff slope
alpha =  3000
C_SHORT = 5000  

C_LOST = 2000
# Connection capacity at fleet bus (MW)
P_CONN = 2

# Pre-terminal buffer parameters
K_BUFFER = 7
DELTA_E  = 0.15
# Value of remaining energy at end of horizon (€/MWh)
V_E_CARRY = 50.0   


# Voltage bounds (LinDistFlow)
V_MIN = 0.95
V_MAX = 1.05

E_MIN_FRAC = 0.5

# Soft target reformulation (minimum required fraction of the target)
TARGET_MIN_FRAC = 0.7   # guarantee 70% of Etarget as a hard minimum
SOC_MIN_DEP = 0.50   # hard minimum departure SoC




# SoC penalty
SOC_PTS  = [0.0, 0.4, 0.7, 1.0]
CDEF_PTS = [400.0, 220.0, 90.0, 50.0]  # convex decreasing

# Scale factor for penalty level (keep if your PDF scales it)
CDEF_SCALE = 50.0

#INPUT DATA

bev      = pd.read_excel(BEV_XLSX)
baseline = pd.read_excel(BASELINE_XLSX)

bev["charger_id"]      = bev["charger_id"].astype(str)
baseline["charger_id"] = baseline["charger_id"].astype(str)

# Time set from baseline (0..23)
T  = sorted(int(t) for t in baseline["t"].unique())
T0 = T[0]
H  = len(T)

print("Hours:", T)

def shift_time(t, k):
    """Return time in T shifted by k steps (k can be negative)."""
    return T[(T.index(t) + k) % H]


#ARRIVAL / DEPARTURE TIMES AND AVAILABILITY
t_arr_map = {}
t_dep_map = {}

for _, row in bev.iterrows():
    b = str(row["charger_id"])
    t_arr_map[b] = int(row["t_arr"]) % 24
    t_dep_map[b] = int(row["t_dep"]) % 24

def availability(ta, td, t):
    """Binary availability of a truck in hour t (wrap-around)."""
    if ta < td:
        return 1 if (ta <= t < td) else 0
    else:
        return 1 if (t >= ta or t < td) else 0

# last charging hour per truck: t_last[b] = (t_dep[b]-1) mod 24
t_last_map = {b: (t_dep_map[b] - 1) % 24 for b in t_dep_map.keys()}


#TIME-SERIES FOR NETWORK: LOAD, PV, NETWORK, SLACK & FLEET BUS

df_load = pd.read_excel(TIMESERIES_XLSX, sheet_name="load_profile")


df_load.rename(columns={df_load.columns[0]: "time_str"}, inplace=True)
df_load = df_load[df_load["time_str"].notna()]

# Convert time to hour 
def parse_hour(x):
    # If already numeric (0–23)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    # If string like "07:00"
    x = str(x).strip()
    return int(x.split(":")[0])

df_load["t"] = df_load["time_str"].apply(parse_hour)
df_load["t"] = df_load["t"].astype(int)

bus_cols = [c for c in df_load.columns if c not in ["time_str", "t"]]

df_load_long = df_load.melt(
    id_vars=["t"],
    value_vars=bus_cols,
    var_name="bus",
    value_name="P_load_pu"
)
df_load_long["bus"] = df_load_long["bus"].astype(int)
df_load_long["t"]   = df_load_long["t"].astype(int)

df_pv = pd.read_excel(TIMESERIES_XLSX, sheet_name="PV_profile")
if "bus_id" in df_pv.columns:
    df_pv.rename(columns={"bus_id": "bus"}, inplace=True)
df_pv["bus"] = df_pv["bus"].astype(int)
df_pv["t"]   = df_pv["t"].astype(int)

net = pd.read_excel(TIMESERIES_XLSX, sheet_name="network")
net["line_id"]  = net["line_id"].astype(int)
net["from_bus"] = net["from_bus"].astype(int)
net["to_bus"]   = net["to_bus"].astype(int)

slack_bus = int(pd.read_excel(TIMESERIES_XLSX, sheet_name="slack_bus").iloc[0, 0])
fleet_bus = int(pd.read_excel(TIMESERIES_XLSX, sheet_name="fleet_bus").iloc[0, 0])

print("Bus set will be built from network:")

I = sorted(set(net["from_bus"]).union(set(net["to_bus"])))
line_ids = list(net["line_id"])

from_bus_map = {int(r["line_id"]): int(r["from_bus"]) for _, r in net.iterrows()}
to_bus_map   = {int(r["line_id"]): int(r["to_bus"])   for _, r in net.iterrows()}
R_map        = {int(r["line_id"]): float(r["R"])      for _, r in net.iterrows()}
X_map        = {int(r["line_id"]): float(r["X"])      for _, r in net.iterrows()}
Smax_map     = {int(r["line_id"]): float(r["S_max"])  for _, r in net.iterrows()}

P_load_dict = {(i, t): 0.0 for i in I for t in T}
PV_dict     = {(i, t): 0.0 for i in I for t in T}

for _, r in df_load_long.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        P_load_dict[(i, t)] = float(r["P_load_pu"])

for _, r in df_pv.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        PV_dict[(i, t)] = float(r["PV"])


#Wholesale price only for plotting (NOT in LL for now)

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
    # keep zeros if sheet missing
    pass


#AGGREGATE THE WHOLE FLEET INTO ONE AGENT
B = list(bev["charger_id"].unique())

def bev_scalar(b, col):
    return float(bev.loc[bev["charger_id"] == b, col].iloc[0])

Pmax_fleet = 1.3 * sum(
    bev.loc[bev["charger_id"] == b, "p_max"].iloc[0]
    for b in B
)

Emax_fleet = sum(bev_scalar(b, "battery_size") for b in B)

E0_fleet = sum(
    bev_scalar(b, "e0") * bev_scalar(b, "battery_size")
    for b in B
)

Etarget_fleet = sum(
    bev_scalar(b, "e_target") * bev_scalar(b, "battery_size")
    for b in B
)

eta_fleet = float(bev["eta"].mean())

# Aggregate availability: fleet is available if any truck is available
a_fleet = {t: int(any(availability(t_arr_map[b], t_dep_map[b], t) for b in B)) for t in T}

# definition for last charging hour (latest among trucks)
t_last_fleet = min((t_dep_map[b] - 1) % 24 for b in B)


Etarget_min_fleet = TARGET_MIN_FRAC * Etarget_fleet

def downstream_buses(start_bus):
    visited = set()
    stack = [start_bus]
    while stack:
        b = stack.pop()
        for _, r in net.iterrows():
            if r["from_bus"] == b and r["to_bus"] not in visited:
                visited.add(r["to_bus"])
                stack.append(r["to_bus"])
    return visited


sigma_baseline_dict = {}

for l in line_ids:
    fb = from_bus_map[l]
    tb = to_bus_map[l]

    # buses downstream of this line
    buses_down = downstream_buses(tb)

    for t in T:
        baseline_flow = sum(
            P_load_dict[(i, t)] - PV_dict[(i, t)]
            for i in buses_down
        )

        sigma_baseline_dict[(l, t)] = max(
            0.0,
            abs(baseline_flow) - Smax_map[l]
        )



#PYOMO MODEL (UL sets λ, LL responds)

m = pyo.ConcreteModel()
m.T = pyo.Set(initialize=T, ordered=True)

#UL decision variable: tariff
m.lambda_t = pyo.Var(m.T, bounds=(LAMBDA_MIN, LAMBDA_MAX))

#For Diagnostics  (NOT in LL objective for now)
m.pi = pyo.Param(m.T, initialize=price_dict, mutable=False)




#LOWER-LEVEL PRIMAL (AGGREGATE FLEET)

m.P = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # fleet charging power
m.E = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # fleet energy

# Shortfall to full target (soft)
m.s_dep = pyo.Var(domain=pyo.NonNegativeReals)  # shortfall at departure (MWh)

m.SoC_dep = pyo.Var(bounds=(0.0, 1.0))           # departure SoC proxy
#m.cdef    = pyo.Var(domain=pyo.NonNegativeReals) # penalty var (€/MWh)

#link to network injection at fleet bus
m.Pfleet = pyo.Var(m.T, domain=pyo.NonNegativeReals)
m.shortfall = pyo.Var(domain=pyo.NonNegativeReals)



#LOWER-LEVEL CONSTRAINTS (AGGREGATE FLEET)

m.a = pyo.Param(m.T, initialize=a_fleet, within=pyo.Binary)

def P_limit(m, t):
    return m.P[t] <= m.a[t] * Pmax_fleet

m.P_LIMIT = pyo.Constraint(m.T, rule=P_limit)




def soc_dyn(m, t):
    if t == T0:
        return m.E[t] == E0_fleet
    t_prev = T[T.index(t) - 1]
    return m.E[t] == m.E[t_prev] + eta_fleet * m.P[t] * DELTA_T
m.SOC = pyo.Constraint(m.T, rule=soc_dyn)

def E_bound(m, t):
    return m.E[t] <= Emax_fleet
m.E_MAX = pyo.Constraint(m.T, rule=E_bound)

m.SOC_DEP_DEF = pyo.Constraint(expr=m.SoC_dep == m.E[t_last_fleet] / Emax_fleet)


t_buf_fleet = shift_time(t_last_fleet, -K_BUFFER)


def pfleet_link(m, t):
    return m.Pfleet[t] == m.P[t]
m.PFLEET_LINK = pyo.Constraint(m.T, rule=lambda m,t: m.Pfleet[t] == m.P[t])


m.TERMINAL_SOFT = pyo.Constraint(
    expr = m.E[t_last_fleet] + m.s_dep >= Etarget_fleet
)

m.SOC_MIN_DEP = pyo.Constraint(
    expr = m.E[t_last_fleet] >= SOC_MIN_DEP * Emax_fleet
)

m.TERMINAL_SOFT = pyo.Constraint(expr = m.E[t_last_fleet] + m.s_dep >= Etarget_fleet)



#NETWORK MODEL (LinDistFlow) – UL

if USE_NETWORK:
    m.I = pyo.Set(initialize=I)
    m.L = pyo.Set(initialize=line_ids)
    m.P_conn = pyo.Param(initialize=P_CONN)

    m.Pload = pyo.Param(m.I, m.T, initialize=P_load_dict, within=pyo.NonNegativeReals)
    m.PPV   = pyo.Param(m.I, m.T, initialize=PV_dict,     within=pyo.NonNegativeReals)

    m.R    = pyo.Param(m.L, initialize=R_map)
    m.X    = pyo.Param(m.L, initialize=X_map)
    m.Smax = pyo.Param(m.L, initialize=Smax_map, within=pyo.NonNegativeReals)

    # Variables
    m.Pij    = pyo.Var(m.L, m.T, domain=pyo.Reals)
    m.Qij    = pyo.Var(m.L, m.T, domain=pyo.Reals)
    m.V      = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

    m.ENS    = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.PVcurt = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.Pslack = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.xi_low  = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.xi_high = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

    m.shortfall = pyo.Var(domain=pyo.NonNegativeReals)
    




    # thermal slack
    m.sigma = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)

    # baseline congestion parameter
   

    m.sigma_baseline = pyo.Param(
        m.L, m.T,
        initialize=sigma_baseline_dict,
        within=pyo.NonNegativeReals,
        mutable=False
    )


    # adjacency
    N_in  = defaultdict(list)
    N_out = defaultdict(list)
    for _, r in net.iterrows():
        lid = int(r["line_id"])
        i   = int(r["from_bus"])
        j   = int(r["to_bus"])
        N_out[i].append(lid)
        N_in[j].append(lid)

    # Nodal active power balance:
    # inflow - outflow = load - ENS - (PV - PVcurt) + fleet - slack
    def nodal_balance(m, i, t):
        inflow  = sum(m.Pij[l, t] for l in N_in[i])  if i in N_in  else 0.0
        outflow = sum(m.Pij[l, t] for l in N_out[i]) if i in N_out else 0.0

        load  = m.Pload[i, t]
        pv    = m.PPV[i, t]
        fleet = m.Pfleet[t] if i == fleet_bus else 0.0
        slack = m.Pslack[t] if i == slack_bus else 0.0

        return inflow - outflow == (load - m.ENS[i, t] - (pv - m.PVcurt[i, t]) + fleet - slack)

    m.PBAL = pyo.Constraint(m.I, m.T, rule=nodal_balance)

    #
    #  bounds
    m.ENS_BND = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.ENS[i,t] <= m.Pload[i,t])
    m.PV_BND  = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.PVcurt[i,t] <= m.PPV[i,t])

    # Voltage drops
    def v_drop(m, l, t):
        i = from_bus_map[int(l)]
        j = to_bus_map[int(l)]
        return m.V[j, t] == m.V[i, t] - 2.0 * (m.R[l] * m.Pij[l, t] + m.X[l] * m.Qij[l, t])
    m.VDROP = pyo.Constraint(m.L, m.T, rule=v_drop)

    

    m.V_LOW  = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.V[i,t] >= V_MIN - m.xi_low[i,t])
    m.V_HIGH = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.V[i,t] <= V_MAX + m.xi_high[i,t])

    m.SLACK_V = pyo.Constraint(m.T, rule=lambda m,t: m.V[slack_bus, t] == 1.0)

    # thermal box limits with slack sigma
    m.P_LIM_UP = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: m.Pij[l,t] <=  m.Smax[l] + m.sigma[l,t])
    m.P_LIM_LO = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: m.Pij[l,t] >= -m.Smax[l] - m.sigma[l,t])
    m.Q_LIM    = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: pyo.inequality(-m.Smax[l], m.Qij[l,t], m.Smax[l]))

    # fleet-caused congestion proxy
    # incremental overload beyond baseline
    m.sigma_inc = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)

    


    # CONGESTION-TRIGGERED TARIFF
   

    #Binary indicator: congestion present at time t
    m.z_cong = pyo.Var(m.T, domain=pyo.Binary)

    #Sum of incremental congestion
    def sigma_sum(m, t):
        return sum(m.sigma_inc[l, t] for l in m.L)

    #Big-M for congestion (upper bound)
    M_CONG = sum(m.Smax[l] for l in m.L) 

    #If no congestion → z_cong = 0
    m.CONG_FLAG = pyo.Constraint(
       m.T,
    rule=lambda m, t: sigma_sum(m, t) <= M_CONG * m.z_cong[t]
    )

    # Minimum tariff when congestion occurs
    LAMBDA_CONG_MIN = 2000.0   # € / MWh

    m.LAM_IF_CONG = pyo.Constraint(
        m.T,
    rule=lambda m, t: m.lambda_t[t] >= LAMBDA_CONG_MIN * m.z_cong[t]
    )


    m.SIG_INC_1 = pyo.Constraint(m.L, m.T,
    rule=lambda m,l,t: m.sigma_inc[l,t] >= m.sigma[l,t] - m.sigma_baseline[l,t])

    m.SIG_INC_2 = pyo.Constraint(m.L, m.T,
    rule=lambda m,l,t: m.sigma_inc[l,t] >= 0.0)

    m.SIG_INC_3 = pyo.Constraint(m.L, m.T,
    rule=lambda m,l,t: m.sigma_inc[l,t] <= m.sigma[l,t])  # prevents inventing inc overload


    
    #CONGESTION → TARIFF LOGIC 

    # Binary congestion indicator
    m.z_cong = pyo.Var(m.T, domain=pyo.Binary)

    # Sum of incremental congestion
    def s_sum(m, t):
        return sum(m.sigma_inc[l, t] for l in m.L)

    # Big-M: valid upper bound on congestion
    M_CONG = sum(Smax_map[l] for l in line_ids)

    # Minimum tariff when congestion exists
    LAMBDA_MIN_CONG = 300.0   # €/MWh

    # If congestion > 0 → z_cong = 1
    m.CONG_LINK = pyo.Constraint(
       m.T,
       rule=lambda m, t: s_sum(m, t) <= M_CONG * m.z_cong[t]
    )

    # If z_cong = 1 → tariff must be active
    m.LAM_MIN_IF_CONG = pyo.Constraint(
       m.T,
       rule=lambda m, t: m.lambda_t[t] >= LAMBDA_MIN_CONG * m.z_cong[t]
    )


    #congestion-driven tariff

    #m.TARIFF_RULE = pyo.Constraint(
    #m.T,
    #rule=lambda m,t: m.lambda_t[t] >= alpha * sum(m.sigma_inc[l,t] for l in m.L)
    #)


    m.SHORTFALL_DEF = pyo.Constraint(
    expr=m.shortfall >= Etarget_fleet - m.E[t_last_fleet]
    )








    # connection capacity
    m.CONN_CAP = pyo.Constraint(m.T, rule=lambda m,t: m.Pfleet[t] <= m.P_conn)

    # tariff ramping
    def ramp_up(m, t):
        if t == T0:
            return pyo.Constraint.Skip
        t_prev = T[T.index(t) - 1]
        return m.lambda_t[t] - m.lambda_t[t_prev] <= DELTA_LAMBDA_MAX

    def ramp_dn(m, t):
        if t == T0:
            return pyo.Constraint.Skip
        t_prev = T[T.index(t) - 1]
        return m.lambda_t[t_prev] - m.lambda_t[t] <= DELTA_LAMBDA_MAX




    M_CONG = sum(Smax_map[l] for l in line_ids)  # safe upper bound
 


    def cong_indicator(m, t):
        return sum(m.sigma_inc[l, t] for l in m.L) <= M_CONG * m.z_cong[t]

    m.CONG_INDICATOR = pyo.Constraint(m.T, rule=cong_indicator)


    #def tariff_activation(m, t):
    #    return m.lambda_t[t] >= LAMBDA_MIN_CONG * m.z_cong[t]

    #m.TARIFF_RULE = pyo.Constraint(m.T, rule=tariff_activation)







# KKT EMBEDDING - Aggregate Fleet
if USE_KKT:
    # Big-M
    M_PRIMAL_P = float(Pmax_fleet)
    M_PRIMAL_E = float(Emax_fleet)
    M_PRIMAL_S = float(Etarget_fleet) + float(Emax_fleet)
    M_DUAL     = 1e4

    # Dual variables
    m.muP_min = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # P[t] >= 0
    m.muP_max = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # P[t] <= a[t] Pmax

    m.muE_min = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # E[t] >= 0
    m.muE_max = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # E[t] <= Emax

    m.nu    = pyo.Var(m.T, domain=pyo.Reals)               # SOC equality dual (free)

    m.gamma = pyo.Var(domain=pyo.NonNegativeReals)  # dual for terminal soft constraint
    m.delta = pyo.Var(domain=pyo.NonNegativeReals)  # dual for s_dep >= 0
    m.epsS  = pyo.Var(domain=pyo.NonNegativeReals)  # stationarity relaxation for s_dep
    # s_dep >= 0 dual
    # dual for hard departure SoC constraint: E[t_last] >= SOC_MIN_DEP * Emax_fleet
    m.muSOC_min = pyo.Var(domain=pyo.NonNegativeReals)

    # binary for complementarity of (E[t_last] - Emin) ⟂ muSOC_min
    m.zSOCmin = pyo.Var(domain=pyo.Binary)


    # Relaxation slacks
    m.epsP = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.epsE = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.epsS = pyo.Var(domain=pyo.NonNegativeReals)



    m.zShort = pyo.Var(domain=pyo.Binary)
    m.zSdep  = pyo.Var(domain=pyo.Binary)



    eps_penalty = W_EPS * sum(m.epsP[t] + m.epsE[t] for t in m.T) + W_EPS * m.epsS

    Emin_dep = SOC_MIN_DEP * Emax_fleet


    # primal slack form: Emin - E_last <= 0  ->  g >= -M(1-z)
    m.SOCMIN_PRIM = pyo.Constraint(
        expr = (Emin_dep - m.E[t_last_fleet]) >= -M_PRIMAL_E * (1 - m.zSOCmin)
    )

    # dual activation
    m.SOCMIN_DUAL = pyo.Constraint(
        expr = m.muSOC_min <= M_DUAL * m.zSOCmin
    )


    # --- Stationarity wrt P[t] ---
    # LL objective term is: sum_t lambda_t * P[t] * Δt
    def ST_P_lo(m, t):
        idx = T.index(t)
        if idx == 0:
            expr = m.lambda_t[t] + m.muP_max[t] - m.muP_min[t]
        else:
            expr = m.lambda_t[t] - eta_fleet * DELTA_T * m.nu[t] + m.muP_max[t] - m.muP_min[t]
        return expr >= -m.epsP[t]

    def ST_P_hi(m, t):
        idx = T.index(t)
        if idx == 0:
            expr = m.lambda_t[t] + m.muP_max[t] - m.muP_min[t]
        else:
            expr = m.lambda_t[t] - eta_fleet * DELTA_T * m.nu[t] + m.muP_max[t] - m.muP_min[t]
        return expr <= m.epsP[t]

    m.ST_P_LOW  = pyo.Constraint(m.T, rule=ST_P_lo)
    m.ST_P_HIGH = pyo.Constraint(m.T, rule=ST_P_hi)

    # --- Stationarity wrt E[t] ---
    # From SOC equality duals: nu[t] - nu[t+1]
    # Terminal at t_last_fleet: -gamma
    # Buffer at t_buf_fleet:    -theta
    def ST_E_lo(m, t):
        idx = T.index(t)
        nu_next = m.nu[T[idx + 1]] if idx < len(T) - 1 else 0.0

        is_dep  = 1.0 if t == t_last_fleet else 0.0
        is_end  = 1.0 if t == max(T) else 0.0   # t = 23

        expr = (
           m.nu[t] - nu_next
           + m.muE_max[t] - m.muE_min[t]
           + is_dep * m.delta
           - is_dep * m.muSOC_min
           - is_end * V_E_CARRY
        )
        return expr >= -m.epsE[t]


    def ST_E_hi(m, t):
        idx = T.index(t)
        nu_next = m.nu[T[idx + 1]] if idx < len(T) - 1 else 0.0

        is_dep  = 1.0 if t == t_last_fleet else 0.0
        is_end  = 1.0 if t == max(T) else 0.0

        expr = (
           m.nu[t] - nu_next
           + m.muE_max[t] - m.muE_min[t]
           + is_dep * m.delta
           - is_dep * m.muSOC_min
           - is_end * V_E_CARRY
        )
        return expr <= m.epsE[t]




    

    m.ST_E_LOW  = pyo.Constraint(m.T, rule=ST_E_lo)
    m.ST_E_HIGH = pyo.Constraint(m.T, rule=ST_E_hi)



    def ST_S_lo(m):
        return (C_SHORT - m.gamma - m.delta) >= -m.epsS
    def ST_S_hi(m):
        return (C_SHORT - m.gamma - m.delta) <=  m.epsS

    m.ST_S_LOW  = pyo.Constraint(rule=ST_S_lo)
    m.ST_S_HIGH = pyo.Constraint(rule=ST_S_hi)



    def ST_SF_lo(m):
        return C_LOST - m.delta >= -m.epsS

    def ST_SF_hi(m):
        return C_LOST - m.delta <=  m.epsS
    

    m.ST_SF_LOW  = pyo.Constraint(rule=ST_SF_lo)
    m.ST_SF_HIGH = pyo.Constraint(rule=ST_SF_hi)


    m.zShort = pyo.Var(domain=pyo.Binary)

    m.SHORT_PRIM = pyo.Constraint(
    expr=(Etarget_fleet - m.E[t_last_fleet] - m.shortfall)
         >= -M_PRIMAL_E * (1 - m.zShort)
    
    )


    m.SHORT_DUAL = pyo.Constraint(
    expr=m.delta <= M_DUAL * m.zShort
    )


    # terminal constraint: g = Etarget - E_last - s_dep <= 0
    def Short_slack(m):
        g = Etarget_fleet - m.E[t_last_fleet] - m.s_dep
        return g >= -M_PRIMAL_E * (1 - m.zShort)
    m.SHORT_PRIM = pyo.Constraint(rule=Short_slack)
    m.SHORT_DUAL = pyo.Constraint(rule=lambda m: m.gamma <= M_DUAL * m.zShort)

    # s_dep >= 0
    m.SDEP_PRIM = pyo.Constraint(rule=lambda m: -m.s_dep >= -M_PRIMAL_S * (1 - m.zSdep))
    m.SDEP_DUAL = pyo.Constraint(rule=lambda m:  m.delta <=  M_DUAL * m.zSdep)








    # Complementarity (Big-M) with binaries
    m.zP_min = pyo.Var(m.T, domain=pyo.Binary)
    m.zP_max = pyo.Var(m.T, domain=pyo.Binary)
    m.zE_min = pyo.Var(m.T, domain=pyo.Binary)
    m.zE_max = pyo.Var(m.T, domain=pyo.Binary)
    #m.zShort = pyo.Var(domain=pyo.Binary)
    #m.zSdep  = pyo.Var(domain=pyo.Binary)

    # P[t] >= 0
    m.PMIN_PRIM = pyo.Constraint(m.T, rule=lambda m,t: -m.P[t] >= -M_PRIMAL_P * (1 - m.zP_min[t]))
    m.PMIN_DUAL = pyo.Constraint(m.T, rule=lambda m,t:  m.muP_min[t] <=  M_DUAL     * m.zP_min[t])

    # P[t] <= a[t]*Pmax_fleet
    def Pmax_slack(m, t):
        g = m.P[t] - m.a[t] *  Pmax_fleet  # g <= 0
        return g >= -M_PRIMAL_P * (1 - m.zP_max[t])
    m.PMAX_PRIM = pyo.Constraint(m.T, rule=Pmax_slack)
    m.PMAX_DUAL = pyo.Constraint(m.T, rule=lambda m,t: m.muP_max[t] <= M_DUAL * m.zP_max[t])



    # E[t] <= Emax_fleet
    def Emax_slack(m, t):
        g = m.E[t] - Emax_fleet
        return g >= -M_PRIMAL_E * (1 - m.zE_max[t])
    m.EMAX_PRIM = pyo.Constraint(m.T, rule=Emax_slack)
    m.EMAX_DUAL = pyo.Constraint(m.T, rule=lambda m,t: m.muE_max[t] <= M_DUAL * m.zE_max[t])



    m.SHORTFALL_DEF = pyo.Constraint(
    expr=m.shortfall >= Etarget_fleet - m.E[t_last_fleet]
    )







    

    #s_dep >= 0
    #m.SDEP_PRIM = pyo.Constraint(rule=lambda m: -m.s_dep >= -M_PRIMAL_S * (1 - m.zSdep))
    #m.SDEP_DUAL = pyo.Constraint(rule=lambda m:  m.delta <=  M_DUAL     * m.zSdep)


# UPPER-LEVEL OBJECTIVE (DSO congestion management)
def UL_obj(m):

    
    # Network-related costs
    
    if USE_NETWORK:
        volt_cost = sum(
            C_VOLT * (m.xi_low[i, t] + m.xi_high[i, t])
            for i in m.I for t in m.T
        )

        therm_cost = sum(
            C_THERM * m.sigma_inc[l,t] 
            for l in m.L for t in m.T
        )

        fleet_cong_cost = C_FLEET * sum(
            m.Pfleet[t] * m.z_cong[t]
            for t in m.T
        )




        ens_cost = sum(
            C_ENS * m.ENS[i, t]
            for i in m.I for t in m.T
        )

        pvc_cost = sum(
            C_PVCURT * m.PVcurt[i, t]
            for i in m.I for t in m.T
        )

    


    else:
        volt_cost     = 0.0
        therm_cost = 0.0
        fleet_cong_cost = 0.0   
        ens_cost      = 0.0
        pvc_cost      = 0.0
    


    # KKT relaxation penalty
    if USE_KKT:
        W_EPS = 1e8  

        eps_penalty = (
            W_EPS * sum(m.epsP[t] + m.epsE[t] for t in m.T)
            
        )
    else:
        eps_penalty = 0.0
    
    # Total UL objective
    
    return (
        volt_cost
        + ens_cost
        + pvc_cost
        + therm_cost
        + eps_penalty
        +fleet_cong_cost

    )


m.OBJ = pyo.Objective(rule=UL_obj, sense=pyo.minimize)




# BASELINE: Fleet availability check
import matplotlib.pyplot as plt

avail = [int(value(m.a[t])) for t in T]

f"avail={int(value(m.a[t]))}"


plt.figure(figsize=(10,4))
plt.step(T, avail, where="post", linewidth=2)
plt.xlabel("Hour")
plt.ylabel("Fleet availability (0/1)")
plt.title("Fleet availability from input data")
plt.ylim(-0.1, 1.1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# BASELINE: Net load at fleet bus 
fleet_bus_id = fleet_bus  

P_load_fleet_bus = np.array([
    P_load_dict[(fleet_bus_id, t)] for t in T
])

PV_fleet_bus = np.array([
    PV_dict[(fleet_bus_id, t)] for t in T
])

net_load_fleet_bus = P_load_fleet_bus - PV_fleet_bus 

plt.figure(figsize=(10,4))
plt.plot(T, net_load_fleet_bus, linewidth=3, label="Net load (load − PV)")
plt.xlabel("Hour")
plt.ylabel("Power [pu or MW]")
plt.title(f"Baseline net load at fleet bus {fleet_bus_id}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# BASELINE: Downstream power through each line
# (DC-style aggregation)

def downstream_buses(start_bus):
    """Very simple downstream bus set (tree assumed)."""
    visited = set()
    stack = [start_bus]
    while stack:
        b = stack.pop()
        for l in net.itertuples():
            if l.from_bus == b and l.to_bus not in visited:
                visited.add(l.to_bus)
                stack.append(l.to_bus)
    return visited

plt.figure(figsize=(12,6))

for l in line_ids:
    fb = from_bus_map[l]
    tb = to_bus_map[l]

    # downstream buses
    buses_down = downstream_buses(tb)

    P_line = np.array([
        sum(
            P_load_dict[(i, t)] - PV_dict[(i, t)]
            for i in buses_down
        )
        for t in T
    ])

    plt.plot(T, P_line, label=f"Line {l}")

    # thermal limit
    plt.axhline(Smax_map[l], color="red", linestyle="--", alpha=0.3)
    plt.axhline(-Smax_map[l], color="red", linestyle="--", alpha=0.3)

plt.xlabel("Hour")
plt.ylabel("Power")
plt.title("Baseline downstream loading per line (no fleet charging)")
plt.grid(alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


from pyomo.environ import SolverFactory

solver = SolverFactory("gurobi")   # or "cbc", "glpk", etc.



# ---------- BASELINE MODEL ----------
m_base = m.clone()
for t in m_base.T:
    m_base.lambda_t[t].fix(0.0)


# Disable congestion–tariff constraints
if hasattr(m_base, "TARIFF_PROP"):
    m_base.TARIFF_PROP.deactivate()
if hasattr(m_base, "LAM_MIN_IF_CONG"):
    m_base.LAM_MIN_IF_CONG.deactivate()
if hasattr(m_base, "z_cong"):
    m_base.z_cong.fix(0)
if hasattr(m_base, "sigma_inc"):
    for l in m_base.L:
        for t in m_base.T:
            m_base.sigma_inc[l, t].fix(0.0)


# Solve baseline
solver.solve(m_base, tee=True)





# SOLVE 
if __name__ == "__main__":

    solver = pyo.SolverFactory("gurobi")
    #Anchor initial tariff

    m.lambda_t[T0].fix(0.0)


    
    #TARIFF CASE (original model)

    result = solver.solve(m, tee=True)
    print(result)

    term = result.solver.termination_condition
    if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        print("\nNo feasible/optimal solution found.\n")
        raise SystemExit(0)

    print("\nMODEL SOLVED (TARIFF CASE)\n")

    #Save ALL tariff-case outputs
    T_list = list(m.T)

    P_tariff      = np.array([value(m.P[t])        for t in T_list])
    E_tariff      = np.array([value(m.E[t])        for t in T_list])
    Pfleet_tariff = np.array([value(m.Pfleet[t])   for t in T_list])
    lambda_tariff = np.array([value(m.lambda_t[t]) for t in T_list])

    if USE_NETWORK:
        sigma_tar = np.array([sum(value(m.sigma_inc[l, t]) for l in m.L) for t in T_list])
        V_tariff  = np.array([[value(m.V[i, t]) for t in T_list] for i in m.I])
    else:
        sigma_tar = None
        V_tariff  = None

    
    SoC_dep_tariff = value(m.SoC_dep) if hasattr(m, "SoC_dep") else None
    epsP_max_tariff = max(value(m.epsP[t]) for t in m.T) if USE_KKT else None
    epsE_max_tariff = max(value(m.epsE[t]) for t in m.T) if USE_KKT else None

    
    # NO-TARIFF BASELINE
    # Save current tariff rule state
    tariff_rule_was_active = m.TARIFF_RULE.active if hasattr(m, "TARIFF_RULE") else False

    # Deactivate tariff coupling and FIX lambda = 0 
    if hasattr(m, "TARIFF_RULE"):
        m.TARIFF_RULE.deactivate()
    for t in m.T:
        m.lambda_t[t].fix(0.0)

    result_nt = solver.solve(m, tee=False)
    term_nt = result_nt.solver.termination_condition
    if term_nt not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        print("\nNo feasible/optimal no-tariff baseline.\n")
        # restore and exit
        for t in m.T:
            m.lambda_t[t].unfix()
        if hasattr(m, "TARIFF_RULE") and tariff_rule_was_active:
            m.TARIFF_RULE.activate()
        raise SystemExit(0)

    print("\nMODEL SOLVED (NO-TARIFF BASELINE)\n")

    # Save baseline outputs
    P_notariff      = np.array([value(m.P[t])      for t in T_list])
    E_notariff      = np.array([value(m.E[t])      for t in T_list])
    Pfleet_notariff = np.array([value(m.Pfleet[t]) for t in T_list])

    if USE_NETWORK:
        sigma_nt = np.array([sum(value(m.sigma_inc[l, t]) for l in m.L) for t in T_list])
        V_nt     = np.array([[value(m.V[i, t]) for t in T_list] for i in m.I])
    else:
        sigma_nt = None
        V_nt     = None


    # RESTORE MODEL TO TARIFF CASE
    
    for t in m.T:
        m.lambda_t[t].unfix()
    if hasattr(m, "TARIFF_RULE") and tariff_rule_was_active:
        m.TARIFF_RULE.activate()

    
    print("\nAggregate fleet schedule (TARIFF CASE):")
    for k, t in enumerate(T_list):
        print(
            f"t={t:02d}  "
            f"P={P_tariff[k]:8.4f}  "
            f"E={E_tariff[k]:8.4f}  "
            f"lambda={lambda_tariff[k]:8.3f}"
        )

    print("\nDeparture (TARIFF CASE):")
    print("t_last:", t_last_fleet, "t_buf:", t_buf_fleet)
    if SoC_dep_tariff is not None:
        print("SoC_dep:", SoC_dep_tariff)

    if USE_KKT:
        print("\nKKT relaxation diagnostics (TARIFF CASE):")
        print("max epsP:", epsP_max_tariff)
        print("max epsE:", epsE_max_tariff)

    print("\nTariff range (TARIFF CASE):", float(lambda_tariff.min()), float(lambda_tariff.max()))


    # PLOTS 
    import matplotlib.pyplot as plt

    # Tariff case: P and lambda
    plt.figure(figsize=(12, 5))
    plt.step(T_list, P_tariff, where="post", linewidth=2, label="Fleet charging P[t] (tariff case)")
    plt.xlabel("Hour")
    plt.ylabel("Power")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.step(T_list, P_tariff, where="post", linewidth=2, label="P[t] (tariff case)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Fleet power")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(T_list, lambda_tariff, "--o", linewidth=1.5, label="Tariff λ[t] (tariff case)")
    ax2.set_ylabel("Tariff")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.tight_layout()
    plt.show()

    # Comparison: baseline vs tariff (fleet charging)
    plt.figure(figsize=(12, 4))
    plt.step(T_list, P_notariff, where="post", lw=2, label="No tariff", color="black")
    plt.step(T_list, P_tariff, where="post", lw=2, label="With tariff", color="#9467bd")
    plt.xlabel("Hour")
    plt.ylabel("Fleet charging power")
    plt.title("Fleet charging: no tariff vs congestion tariff")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Voltage violation comparison (if network)
    if USE_NETWORK and (V_nt is not None) and (V_tariff is not None):

        def voltage_violation(V):
            return np.maximum(0, V_MIN - V) + np.maximum(0, V - V_MAX)

        viol_nt  = voltage_violation(V_nt).sum(axis=0)
        viol_tar = voltage_violation(V_tariff).sum(axis=0)

        plt.figure(figsize=(12, 4))
        plt.plot(T_list, viol_nt,  lw=2, label="No tariff", color="black")
        plt.plot(T_list, viol_tar, lw=2, label="With tariff", color="#ff7f0e")
        plt.xlabel("Hour")
        plt.ylabel("Voltage violation magnitude")
        plt.title("Voltage violations: no tariff vs tariff")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()




# GENERAL LOAD: BEFORE vs AFTER OPTIMIZATION


# 1) Baseline total system net load (sum over all buses)
baseline_net_load = np.array([
    sum(P_load_dict[(i, t)] - PV_dict[(i, t)] for i in I)
    for t in T_list
])

# 2) Total load BEFORE optimization (no tariff)
total_load_before = baseline_net_load + Pfleet_notariff

# 3) Total load AFTER optimization (with tariff)
total_load_after  = baseline_net_load + Pfleet_tariff

# 4) Plot
plt.figure(figsize=(12, 5))

plt.plot(T_list, total_load_before,
         lw=3, linestyle="--", color="black",
         label="Total load BEFORE optimization (no tariff)")

plt.plot(T_list, total_load_after,
         lw=3, color="#1f77b4",
         label="Total load AFTER optimization (with tariff)")

plt.xlabel("Hour")
plt.ylabel("Total system load [MW or pu]")
plt.title("General system load: congestion resolution via tariff optimization")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()







# -------------------------------------------------
# NOTE:
# Anything below that calls value(m.lambda_t[t]) will reflect the *restored model*
# state (tariff rule active, lambda unfixed), but it will NOT automatically restore
# the tariff-case solution. For plotting and comparisons, keep using the saved arrays.
# -------------------------------------------------


