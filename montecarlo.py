import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime

num_runs = 10000
num_years = 21
mean_return = 0.06
market_volatility = 0.17

class Person:
    def __init__(self, fname, lname, dob,retirement_age, ss_age):
        self.fname=fname
        self.lname =lname
        self.dob=dob
        self.retirement_age=retirement_age
        self.ss_age=ss_age

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

Madison=Person("Madison", "Stone", datetime.date(1986,3,9) , 61 ,62)
Greg=Person("Greg", "Stone", datetime.date(1987,2,17), 62 , 62)
DSH_Trust=Trust("David S Huy Trust",datetime.date(2025,6,10))

mads_410k=Account("Madison 401k",Madison,56000,12000,"tax deferred",True)
greg_roth=Account("Greg Roth",Greg,55000,7000,"after tax",True)
greg_ira=Account("Greg IRA",Greg,172000,0,"tax deferred",True)

trust_brokerage=Account("Trust Brokerage", DSH_Trust, 360000,0,"taxable",False)
trust_IRA=Account("Trust IRA", DSH_Trust, 55000,0,"tax deferred",True)
trust_Roth=Account("Trust Roth", DSH_Trust, 55000,0,"after-tax",True)

accounts=mads_410k,greg_roth,greg_ira,trust_brokerage,trust_IRA,trust_Roth




# Lognormal drift adjusted for variance (Ito correction)
drift = mean_return - 0.5 * market_volatility ** 2

returns = np.random.lognormal(
    mean=drift,
    sigma=market_volatility,
    size=(num_years, num_runs)
)

RMD_Factor={
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

def simulate_account(account, returns,RMD_Factor,num_simulations):
    if type(account.owner)==Trust and account.is_retirement:
        years=account.owner.ret_accnt_end_date.year-account.owner.start_date.year
    else:
        years, num_simulations = returns.shape
    values = np.zeros((years, num_simulations))
    RMD_income = np.zeros((years, num_simulations))
    balance = np.full(num_simulations, float(account.start_balance))
    
    if account.tax_treatment=="tax deferrred":
        for year in range(years):
            age=(year + (today()-account.owner.dob).year)
            if age <= account.owner.retirement_age:
                balance = balance * returns[year] + account.annual_contribution
                values[year] = balance
                RMD_income[year]=0
            elif age >= 75:
                RMD_income[year]=balance/RMD_Factor[age]
                balance = balance * returns[year] - balance/RMD_Factor[age]
                values[year] = balance
            else:
                balance = balance * returns[year]
                values[year] = balance
                RMD_income[year]=balance/RMD_Factor[age]
    else:
        for year in range(years):
            balance = balance * returns[year] + account.annual_contribution
            values[year] = balance
    return values

def get_percentile(account_values, percentile):
    return np.percentile(account_values[-1], percentile)

def plot_percentiles(account_values, num_years, start_year=2026, selected_percentile=50):
    years = np.arange(start_year, start_year + num_years)
    p10  = np.percentile(account_values, 10,  axis=1)
    p25  = np.percentile(account_values, 25,  axis=1)
    p50  = np.percentile(account_values, 50,  axis=1)
    p75  = np.percentile(account_values, 75,  axis=1)
    p90  = np.percentile(account_values, 90,  axis=1)
    psel = np.percentile(account_values, selected_percentile, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(years, p10, p90, alpha=0.2, color="steelblue", label="10th–90th")
    ax.fill_between(years, p25, p75, alpha=0.35, color="steelblue", label="25th–75th")
    ax.plot(years, p50,  color="steelblue", linewidth=2, label="Median (50th)")
    ax.plot(years, psel, color="red",       linewidth=2, linestyle="--",
            label=f"{selected_percentile}th percentile")

    ax.set_xlim(start_year, start_year + num_years - 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Balance")
    ax.set_title("Monte Carlo Portfolio Simulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Interactive crosshair + tooltip
    cursor_vline = ax.axvline(color="gray", linewidth=0.8, linestyle="--", visible=False)
    tooltip = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                          fontsize=9)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            cursor_vline.set_visible(False)
            tooltip.set_visible(False)
            fig.canvas.draw_idle()
            return
        idx = int(round(event.xdata)) - start_year
        idx = max(0, min(idx, num_years - 1))
        year = years[idx]
        cursor_vline.set_xdata([year])
        cursor_vline.set_visible(True)
        lines = {
            f"90th": p90[idx], f"75th": p75[idx], f"50th": p50[idx],
            f"25th": p25[idx], f"10th": p10[idx],
            f"{selected_percentile}th": psel[idx],
        }
        text = f"{year}\n" + "\n".join(f"{k}: ${v:,.0f}" for k, v in lines.items())
        tooltip.set_text(text)
        tooltip.xy = (year, event.ydata)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()

account_values = 0
for acct in accounts:
    account_values += simulate_account(acct,returns, RMD_Factor,num_runs)

plot_percentiles(account_values, num_years, start_year=2026, selected_percentile=33)
