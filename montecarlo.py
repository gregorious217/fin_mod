import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime

num_runs = 10000
num_years = 56
mean_return = 0.06
market_volatility = 0.17

class Person:
    def __init__(self, fname, lname, dob, retirement_age, ss_age, ss_pia,has_pension,pension_startdate,salary):
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.retirement_age = retirement_age
        self.ss_age = ss_age
        self.ss_pia = ss_pia
        self.has_pension=has_pension
        self.pension_startdate=pension_startdate
        self.salary=salary

    def calcPension(self):
        serv_years = self.retirement_age - (self.pension_startdate.year - self.dob.year)
        pensionable_income = max(self.salary, 159733)
        return serv_years * 0.02 * pensionable_income

    def calcSS(self, start_year, num_years):
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
        ss_start_year = self.dob.year + self.ss_age

        income = np.zeros(num_years)
        for i, year in enumerate(range(start_year, start_year + num_years)):
            if year >= ss_start_year:
                income[i] = annual_benefit
        return income


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
        self.balances = None    # shape: (years, 1 + num_simulations); col 0 = calendar year
        self.rmd_income = None  # shape: (years, num_simulations)

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
                age = year + (today - self.owner.dob).days // 365
                if age <= self.owner.retirement_age:
                    balance = balance * returns[year] + self.annual_contribution
                    values[year] = balance
                    RMD_income[year] = 0
                elif age >= 75:
                    rmd_key = min(age, 95)
                    RMD_income[year] = balance / RMD_Factor[rmd_key]
                    balance = balance * returns[year] - balance / RMD_Factor[rmd_key]
                    values[year] = balance
                else:
                    balance = balance * returns[year]
                    values[year] = balance
                    RMD_income[year] = 0

        else:
            for year in range(years):
                balance = balance * returns[year] + self.annual_contribution
                values[year] = balance
                RMD_income[year] = 0

        year_col = np.arange(start_year, start_year + years, dtype=float).reshape(-1, 1)
        self.balances   = np.hstack([year_col, values])
        self.rmd_income = RMD_income

Madison=Person("Madison", "Stone", datetime.date(1986,3,9) , 61 ,62,3191,False,None,95000)
Greg=Person("Greg", "Stone", datetime.date(1987,2,17), 62 , 62,4006,True,datetime.date(2018,3,1),160000)
persons = [Madison, Greg]
DSH_Trust=Trust("David S Huy Trust",datetime.date(2025,6,10))

mads_410k=Account("Madison 401k",Madison,56000,12000,"tax deferred",True)
greg_roth=Account("Greg Roth",Greg,55000,7000,"after tax",True)
greg_ira=Account("Greg IRA",Greg,172000,0,"tax deferred",True)

trust_brokerage=Account("Trust Brokerage", DSH_Trust, 360000,0,"taxable",False)
trust_IRA=Account("Trust IRA", DSH_Trust, 55000,0,"tax deferred",True)
trust_Roth=Account("Trust Roth", DSH_Trust, 55000,0,"after-tax",True)

accounts=mads_410k,greg_roth,greg_ira



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


def plot_income(accounts, persons, start_year, num_years, selected_percentile=50):
    years = np.arange(start_year, start_year + num_years)
    income_components = {}

    for person in persons:
        if person.has_pension:
            annual_pension = person.calcPension()
            retirement_year = person.dob.year + person.retirement_age
            pension = np.zeros(num_years)
            for i, yr in enumerate(range(start_year, start_year + num_years)):
                if yr >= retirement_year:
                    pension[i] = annual_pension
            income_components[f"{person.fname} Pension"] = pension

    for person in persons:
        ss = person.calcSS(start_year, num_years)
        if ss.any():
            income_components[f"{person.fname} SS"] = ss

    rmd_total = np.zeros(num_years)
    for acct in accounts:
        if acct.rmd_income is not None and (
            acct.tax_treatment == "tax deferred" or
            (isinstance(acct.owner, Trust) and acct.is_retirement)
        ):
            rmd_pct = np.percentile(acct.rmd_income, selected_percentile, axis=1)
            if rmd_pct.shape[0] < num_years:
                rmd_pct = np.concatenate([rmd_pct, np.zeros(num_years - rmd_pct.shape[0])])
            rmd_total += rmd_pct
    if rmd_total.any():
        income_components[f"RMDs ({selected_percentile}th pct)"] = rmd_total

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(num_years)
    for (label, values), color in zip(income_components.items(), colors):
        ax.bar(years, values, bottom=bottom, label=label, color=color, alpha=0.85, width=0.7)
        bottom += values

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(start_year - 0.5, start_year + num_years - 0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Income")
    ax.set_title("Retirement Income by Year")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Precompute per-year totals for tooltip
    totals = sum(income_components.values())

    tooltip = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                          fontsize=9)
    tooltip.set_visible(False)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            tooltip.set_visible(False)
            fig.canvas.draw_idle()
            return
        idx = int(round(event.xdata)) - start_year
        idx = max(0, min(idx, num_years - 1))
        year = years[idx]
        lines = [f"{year}"]
        for label, values in income_components.items():
            if values[idx] > 0:
                lines.append(f"{label}: ${values[idx]:,.0f}")
        lines.append(f"Total: ${totals[idx]:,.0f}")
        tooltip.set_text("\n".join(lines))
        tooltip.xy = (event.xdata, event.ydata)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()


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

def plot_account_values(accounts, num_years, start_year=2026, selected_percentile=50):
    years = np.arange(start_year, start_year + num_years)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]

    account_lines = {}
    for acct in accounts:
        bal = acct.balances[:, 1:]
        if bal.shape[0] < num_years:
            bal = np.vstack([bal, np.zeros((num_years - bal.shape[0], bal.shape[1]))])
        account_lines[acct.name] = np.percentile(bal, selected_percentile, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, values), color in zip(account_lines.items(), colors):
        ax.plot(years, values, label=name, color=color, linewidth=2)

    ax.set_xlim(start_year, start_year + num_years - 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Account Balance")
    ax.set_title(f"Account Values Over Time ({selected_percentile}th Percentile)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    cursor_vline = ax.axvline(color="gray", linewidth=0.8, linestyle="--", visible=False)
    tooltip = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                          fontsize=9)
    tooltip.set_visible(False)

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
        total = sum(v[idx] for v in account_lines.values())
        lines = [f"{year}"]
        for name, values in account_lines.items():
            if values[idx] > 0:
                lines.append(f"{name}: ${values[idx]:,.0f}")
        lines.append(f"Total: ${total:,.0f}")
        tooltip.set_text("\n".join(lines))
        tooltip.xy = (year, event.ydata)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()


account_values = 0
for acct in accounts:
    acct.simulate(returns, RMD_Factor, start_year=2026)
    bal = acct.balances[:, 1:]  # col 0 is calendar year
    if bal.shape[0] < num_years:
        bal = np.vstack([bal, np.zeros((num_years - bal.shape[0], bal.shape[1]))])
    account_values += bal

plot_percentiles(account_values, num_years, start_year=2026, selected_percentile=33)
plot_account_values(accounts, num_years, start_year=2026, selected_percentile=33)
plot_income(accounts, persons, start_year=2026, num_years=num_years, selected_percentile=33)
