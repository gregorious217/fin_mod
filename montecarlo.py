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

class Account:
    def __init__(self,name,owner,start_balance,annual_contribution,tax_treatment):
        self.name = name
        self.owner=owner
        self.start_balance=start_balance
        self.annual_contribution=annual_contribution
        self.tax_treatment=tax_treatment

Madison=Person("Madison", "Stone", datetime.date(1986,3,9) , 61 ,62)
Greg=Person("Greg", "Stone", datetime.date(1987,2,17), 62 , 62)
Trust=Person("David S Huy Trust",datetime.date(2025,6,10),None,None,None)

mads_410k=Account("Madison 401k",Madison,56000,12000,"tax deferred")
greg_roth=Account("Greg Roth",Greg,55000,7000,"after tax")
greg_ira=Account("Greg IRA",Greg,172000,0,"tax deferred")

trust_brokerage=Account("Trust Brokerage", Trust, 360000,0,"taxable")
trust_IRA=Account("Trust IRA", Trust, 55000,0,"tax deferred")
trust_Roth=Account("Trust Roth", Trust, 55000,0,"after-tax")

accounts=mads_410k,greg_roth,greg_ira




# Lognormal drift adjusted for variance (Ito correction)
drift = mean_return - 0.5 * market_volatility ** 2

returns = np.random.lognormal(
    mean=drift,
    sigma=market_volatility,
    size=(num_years, num_runs)
)

def simulate_account(starting_value, returns, annual_contribution=0):
    num_years, num_runs = returns.shape
    values = np.zeros((num_years, num_runs))
    balance = np.full(num_runs, float(starting_value))
    for year in range(num_years):
        balance = balance * returns[year] + annual_contribution
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
    account_values += simulate_account(acct.start_balance, returns, acct.annual_contribution)

plot_percentiles(account_values, num_years, start_year=2026, selected_percentile=33)
