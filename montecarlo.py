import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime

num_runs = 10000
num_years = 56
mean_return = 0.06
market_volatility = 0.17
selected_percentile=33

class Person:
    def __init__(self, fname, lname, dob, retirement_date, ss_age, ss_pia,has_pension,pension_startdate,salary):
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.retirement_date = retirement_date
        self.ss_age = ss_age
        self.ss_pia = ss_pia
        self.has_pension=has_pension
        self.pension_startdate=pension_startdate
        self.salary=salary
        

    @property
    def age(self):
        today = datetime.date.today()
        # Subtracts 1 if the current calendar day is before the birth calendar day
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

    @property
    def ret_age(self):
        return self.retirement_date.year - self.dob.year - ((self.retirement_date.month, self.retirement_date.day) < (self.dob.month, self.dob.day))
   
    def calcPension(self):
        if self.has_pension:
            self.ret_age
            serv_mos=(self.retirement_date.year-self.pension_startdate.year)*12+(self.retirement_date.month-self.pension_startdate.month)
            serv_years = serv_mos/12
            pensionable_income = max(self.salary, 159733)
            pension_factor={57:.015,58:.016,59:.017,60:.018,61:.019,62:.02,63:.021,64:.022,65:.023,64:.024,65:.025}
            #print(ret_age)
            #print(serv_years)
            return serv_years * pension_factor[self.ret_age]* pensionable_income
        else:
            pass

    def calcSS(self):
        birth_year = self.dob.year
        if birth_year >= 1960:
            fra = 67
        elif birth_year >= 1955:
            fra = 66 + (birth_year - 1954) / 6
        else:
            fra = 66

        months_diff = (self.ss_age - fra) * 12
        if months_diff < 0:
            months_early = abs(months_diff)
            if months_early <= 36:
                factor = 1 - (5 / 9 * 0.01 * months_early)
            else:
                factor = 1 - (5 / 9 * 0.01 * 36) - (5 / 12 * 0.01 * (months_early - 36))
        elif months_diff > 0:
            years_delayed = min(months_diff / 12, 70 - fra)
            factor = 1 + 0.08 * years_delayed
        else:
            factor = 1.0

        annual_benefit = self.ss_pia * 12 * factor * 0.77
        
        return annual_benefit

class Trust:
    def __init__(self, name,start_date):
        self.name = name
        self.start_date=start_date
        self.ret_accnt_end_date=datetime.date(self.start_date.year + 10,12,31)
        
class Account:
    def __init__(self,name,owner,start_balance,annual_contribution,tax_treatment,is_retirement):
        self.name = name
        self.owner=owner
        self.start_balance=start_balance
        self.annual_contribution=annual_contribution
        self.tax_treatment=tax_treatment
        self.is_retirement=is_retirement
        self.balances = None    
        self.rmd_income = None  

    def simulate(self, returns, RMD_Factor, start_year=None):
        if start_year is None:
            start_year = datetime.date.today().year
        num_years, num_simulations = returns.shape

        is_trust_retirement = isinstance(self.owner, Trust) and self.is_retirement

        if is_trust_retirement:
            years = self.owner.ret_accnt_end_date.year - self.owner.start_date.year
        else:
            years = num_years

        values    = np.zeros((years, num_simulations))
        RMD_income = np.zeros((years, num_simulations))
        balance   = np.full(num_simulations, float(self.start_balance))
        today     = datetime.date.today()

        if is_trust_retirement:
            # SECURE Act 10-year rule: inherited retirement account must be fully
            # distributed by end of year 10. Take 1/years_remaining each year;
            # final year sweeps entire remaining balance into RMD income.
            for year in range(years):
                balance = balance * returns[year]
                years_remaining = years - year
                if years_remaining == 1:
                    RMD_income[year] = balance
                    balance = np.zeros(num_simulations)
                else:
                    RMD_income[year] = balance / years_remaining
                    balance = balance - RMD_income[year]
                values[year] = balance

        elif self.tax_treatment == "tax deferred":
            for year in range(years):
                cur_age = self.owner.age + year
                if cur_age + year<= self.owner.ret_age:
                    balance = balance * returns[year] + self.annual_contribution
                    values[year] = balance
                    RMD_income[year] = 0
                elif cur_age >= 75:
                    rmd_key = min(cur_age, 95)
                    RMD_income[year] = balance / RMD_Factor[rmd_key]
                    #print(RMD_income[year])
                    balance = balance * returns[year] - balance / RMD_Factor[rmd_key]
                    values[year] = balance
                else:
                    balance = balance * returns[year]
                    values[year] = balance
                    RMD_income[year] = 0

        else:
            for year in range(years):
                cur_age = self.owner.age + year
                if cur_age <= self.owner.ret_age:
                    balance = balance * returns[year] + self.annual_contribution
                    values[year] = balance
                    RMD_income[year] = 0
                else:
                    balance = balance * returns[year] 
                    values[year] = balance
                    RMD_income[year] = 0

        self.balances   = values
        self.rmd_income = RMD_income

Madison=Person("Madison", "Stone", datetime.date(1986,3,9) , datetime.date(2049,3,10) ,62,3191,False,None,95000)
Greg=Person("Greg", "Stone", datetime.date(1987,2,17), datetime.date(2049,2,18) , 62, 4006,True,datetime.date(2018,3,1),160000)
persons = [Madison, Greg]
DSH_Trust=Trust("David S Huy Trust",datetime.date(2025,6,10))

mads_410k=Account("Madison 401k",Madison,56000,12000,"tax deferred",True)
greg_roth=Account("Greg Roth",Greg,55000,7000,"after tax",True)
greg_ira=Account("Greg IRA",Greg,172000,0,"tax deferred",True)

trust_brokerage=Account("Trust Brokerage", DSH_Trust, 360000,0,"taxable",False)
trust_IRA=Account("Trust IRA", DSH_Trust, 55000,0,"tax deferred",True)
trust_Roth=Account("Trust Roth", DSH_Trust, 55000,0,"after-tax",True)

accounts=mads_410k,greg_roth,greg_ira


def build_pension_array(user):
    pensionarray=np.zeros((num_years,num_runs))
    today=datetime.date.today()
    for year in range(num_years):
        if today.year+year >= user.retirement_date.year:
            pensionarray[year]=user.calcPension()
        else:
            pensionarray[year]=0
    return pensionarray

def build_ss_array(user):
    ss_array=np.zeros((num_years,num_runs))
    today=datetime.date.today()
    for year in range(num_years):
        if user.age+year >= user.ss_age:
            ss_array[year]=user.calcSS()
        else:
            ss_array[year]=0
    return ss_array

# Lognormal drift adjusted for variance (Ito correction)
drift = mean_return - 0.5 * market_volatility ** 2

returns = np.random.lognormal(
    mean=drift,
    sigma=market_volatility,
    size=(num_years, num_runs)
)

RMD_Factor={
    40:45.7,
    41:44.8,
    42:43.8,
    43:42.9,
    44:41.9,
    45:41.0,
    46:40.0,
    47:39.0,
    48:37.1,
    50:36.2,
    75:24.6,
    76:23.7,
    77:22.9,
    78:22.0,
    79:21.1,
    80:20.2,
    81:19.4,
    82:18.5,
    83:17.7,
    84:16.8,
    85:16.0,
    86:15.2,
    87:14.4,
    88:13.7,
    89:12.9,
    90:12.2,
    91:11.5,
    92:10.8,
    93:10.1,
    94:9.5,
    95:8.9
}

gregPension=build_pension_array(Greg)
gregSS=build_ss_array(Greg)
madisonSS=build_ss_array(Madison)

mads_410k.simulate(returns,RMD_Factor,None)
greg_ira.simulate(returns,RMD_Factor,None)
greg_roth.simulate(returns,RMD_Factor,None)

retirement_spend=np.full(num_years,120000/0.7)
portfolio_withdrawal=np.zeros(num_years)

greg_roth_percentile=np.percentile(greg_roth.balances,selected_percentile,axis=1)

for year in range(num_years):
    if Greg.age+year>=Greg.ret_age:
        total_fixed_income=np.percentile(gregPension,selected_percentile,axis=1)+ np.percentile(gregSS,selected_percentile,axis=1)+np.percentile(madisonSS,selected_percentile,axis=1)+np.percentile(mads_410k.rmd_income,selected_percentile,axis=1)+np.percentile(greg_ira.rmd_income,selected_percentile,axis=1)
        portfolio_withdrawal[year]=max(0,retirement_spend[year]-total_fixed_income[year])
        greg_roth_percentile[year]=greg_roth_percentile[year] - portfolio_withdrawal[year]
    else:
        portfolio_withdrawal[year]=0
        greg_roth_percentile[year]=greg_roth_percentile[year]




COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#F8F8F8")
ax.set_facecolor("#F8F8F8")

years = np.arange(datetime.date.today().year, datetime.date.today().year + num_years)

p = lambda arr: np.percentile(arr, selected_percentile, axis=1)

layers = [
    (p(madisonSS),            "Madison SS"),
    (p(gregSS),               "Greg SS"),
    (p(gregPension),          "Greg Pension"),
    (p(mads_410k.rmd_income), "Madison RMDs"),
    (p(greg_ira.rmd_income),  "Greg RMDs"),
    (portfolio_withdrawal,    "Portfolio Withdrawal"),
]

# Store bars and per-layer values for tooltip lookup
bar_containers = []
layer_values   = []
bottom = np.zeros(num_years)
for (values, label), color in zip(layers, COLORS):
    bars = ax.bar(years, values, bottom=bottom, label=label,
                  color=color, edgecolor="white", linewidth=0.4)
    bar_containers.append(bars)
    layer_values.append(values.copy())
    bottom += values

spend_line = ax.axhline(y=120000 / 0.7, linestyle=":", color="#E84040",
                        linewidth=1.8, label="Retirement Spending Target")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Annual Income / Withdrawal ($)", fontsize=11)
ax.set_title(f"Projected Income Stack — {selected_percentile}th Percentile", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.85, fontsize=9)
ax.grid(axis="y", color="white", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)

# ── Hover tooltip ──────────────────────────────────────────────────────────
cursor_vline = ax.axvline(color="gray", linewidth=0.8, linestyle="--", visible=False)
tooltip = ax.annotate(
    "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#AAAAAA", alpha=0.95),
    fontsize=9, family="monospace",
)

def on_move(event):
    if event.inaxes != ax or event.xdata is None:
        cursor_vline.set_visible(False)
        tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return

    yr = int(round(event.xdata))
    idx = yr - years[0]
    if idx < 0 or idx >= num_years:
        cursor_vline.set_visible(False)
        tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return

    cursor_vline.set_xdata([yr])
    cursor_vline.set_visible(True)

    lines = [f"  {yr}"]
    total = 0.0
    for (_, label), vals in zip(layers, layer_values):
        v = vals[idx]
        if v > 0:
            lines.append(f"  {label:<22} ${v:>10,.0f}")
            total += v
    lines.append(f"  {'─'*34}")
    lines.append(f"  {'Total':<22} ${total:>10,.0f}")

    tooltip.set_text("\n".join(lines))
    tooltip.xy = (yr, event.ydata)
    # Flip tooltip to left side when near right edge
    x_frac = (yr - years[0]) / num_years
    tooltip.set_position((-160, 12) if x_frac > 0.75 else (12, 12))
    tooltip.set_visible(True)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.tight_layout()
plt.show()


