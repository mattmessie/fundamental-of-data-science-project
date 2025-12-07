import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================================
# LOAD DATA
# ==========================================================

data_2019 = pd.read_csv("2019data7.csv")
data_2022 = pd.read_csv("2022data7.csv")

data_2019["Date"] = pd.to_datetime(data_2019["Date"])
data_2022["Date and time"] = pd.to_datetime(data_2022["Date and time"])
data_2022["Date"] = pd.to_datetime(data_2022["Date and time"].dt.date)

# ==========================================================
# CREATE DAILY TOTAL PASSENGERS
# ==========================================================

# Total passengers per day in 2019 (bus + tram + metro)
data_2019["Total passengers"] = (
    data_2019["Bus pax number peak"]   + data_2019["Bus pax number offpeak"] +
    data_2019["Tram pax number peak"]  + data_2019["Tram pax number offpeak"] +
    data_2019["Metro pax number peak"] + data_2019["Metro pax number offpeak"]
)

daily_passengers_2019 = (
    data_2019[["Date", "Total passengers"]]
    .groupby("Date")
    .sum()
    .reset_index()
)
daily_passengers_2019["DayOfYear"] = daily_passengers_2019["Date"].dt.dayofyear
daily_passengers_2019 = daily_passengers_2019.sort_values("DayOfYear")

# Total passengers per day in 2022 = number of sample journeys each day
daily_passengers_2022 = (
    data_2022
    .groupby("Date")
    .size()
    .reset_index(name="Total passengers")
)
daily_passengers_2022["DayOfYear"] = daily_passengers_2022["Date"].dt.dayofyear
daily_passengers_2022 = daily_passengers_2022.sort_values("DayOfYear")

# ==========================================================
# FOURIER SMOOTHING (8 TERMS)
# ==========================================================

def fourier_smooth(values, n_terms=8):
    fft_vals = np.fft.fft(values)
    filtered = np.zeros_like(fft_vals)
    filtered[:n_terms] = fft_vals[:n_terms]
    return np.fft.ifft(filtered).real

smoothed_2019 = fourier_smooth(daily_passengers_2019["Total passengers"].values, 8)
smoothed_2022 = fourier_smooth(daily_passengers_2022["Total passengers"].values, 8)

# ==========================================================
# FIGURE 1 – DAILY PASSENGERS WITH FOURIER SMOOTHING
# ==========================================================

plt.figure(figsize=(12, 6))

plt.scatter(
    daily_passengers_2019["DayOfYear"],
    daily_passengers_2019["Total passengers"],
    s=12, alpha=0.7, label="2019 (scatter)"
)

plt.scatter(
    daily_passengers_2022["DayOfYear"],
    daily_passengers_2022["Total passengers"],
    s=12, alpha=0.7, label="2022 (scatter)"
)

plt.plot(
    daily_passengers_2019["DayOfYear"],
    smoothed_2019,
    linewidth=2.5, label="2019 Fourier 8-term"
)

plt.plot(
    daily_passengers_2022["DayOfYear"],
    smoothed_2022,
    linewidth=2.5, label="2022 Fourier 8-term"
)

plt.xlabel("Day of Year (1–365)")
plt.ylabel("Total Daily Passengers")
plt.title("Figure 1 – Daily Passenger Numbers (2019 vs 2022)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.figtext(0.99, 0.02, "Student ID: 24136779", ha="right")
plt.tight_layout()
plt.savefig("Figure1.png", dpi=300)
plt.show()

# ==========================================================
# PREP FOR FIGURE 2 – WEEKDAY AVERAGES & REVENUES
# ==========================================================

daily_2019 = daily_passengers_2019.copy()
daily_2022 = daily_passengers_2022.copy()

daily_2019["Weekday"] = daily_2019["Date"].dt.day_name()
daily_2022["Weekday"] = daily_2022["Date"].dt.day_name()

avg_2019 = daily_2019.groupby("Weekday")["Total passengers"].mean()
avg_2022 = daily_2022.groupby("Weekday")["Total passengers"].mean()

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
avg_2019 = avg_2019.reindex(weekday_order)
avg_2022 = avg_2022.reindex(weekday_order)

# Revenues (X, Y, Z)
X = data_2019["Bus revenue peak"].sum()   + data_2019["Bus revenue offpeak"].sum()
Y = data_2019["Tram revenue peak"].sum()  + data_2019["Tram revenue offpeak"].sum()
Z = data_2019["Metro revenue peak"].sum() + data_2019["Metro revenue offpeak"].sum()

print(f"X (Bus revenue 2019)   = €{X:,.0f}")
print(f"Y (Tram revenue 2019)  = €{Y:,.0f}")
print(f"Z (Metro revenue 2019) = €{Z:,.0f}")

# ==========================================================
# FIGURE 2 – AVERAGE PASSENGERS PER WEEKDAY (LOG SCALE)
# ==========================================================

x = np.arange(len(weekday_order))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(x - width/2, avg_2019.values, width, label="2019")
plt.bar(x + width/2, avg_2022.values, width, label="2022")

plt.xlabel("Day of Week")
plt.ylabel("Average Number of Passengers per Day (log scale)")
plt.xticks(x, weekday_order)
plt.title("Figure 2 – Average Passengers per Weekday (2019 vs 2022)")
plt.yscale("log")
plt.legend(loc="upper left")

# --- DISPLAY X, Y, Z ON FIGURE (REQUIRED BY ASSIGNMENT) ---
text_xyz = (
    f"X (Bus revenue 2019):  €{X:,.0f}\n"
    f"Y (Tram revenue 2019): €{Y:,.0f}\n"
    f"Z (Metro revenue 2019): €{Z:,.0f}"
)

plt.gca().text(
    0.02, 0.98, text_xyz,
    transform=plt.gca().transAxes,
    va="top", ha="left",
    bbox=dict(boxstyle="round", alpha=0.2)
)
# -----------------------------------------------------------

plt.figtext(0.99, 0.02, "Student ID: 24136779", ha="right")

plt.tight_layout()
plt.savefig("Figure2.png", dpi=300)
plt.show()

# ==========================================================
# FIGURE 3 – METRO PRICE vs DISTANCE WITH REGRESSION
# ==========================================================

metro_2022 = data_2022[data_2022["Mode"].str.lower() == "metro"]

if len(metro_2022) > 0:
    x_dist = metro_2022["Distance"].values
    y_price = metro_2022["Price"].values

    slope, intercept, r, p, se = stats.linregress(x_dist, y_price)

    line_x = np.linspace(x_dist.min(), x_dist.max(), 200)
    line_y = slope * line_x + intercept

    plt.figure(figsize=(10, 6))

    plt.scatter(x_dist, y_price, alpha=0.6, label="Metro journeys", color="teal")
    plt.plot(line_x, line_y, linewidth=2.5, label="Regression", color="darkred")

    plt.xlabel("Distance (km)")
    plt.ylabel("Price (€)")
    plt.title("Figure 3 – Metro Price vs Distance (2022)")
    plt.legend()

    eqn = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r**2:.3f}"
    plt.gca().text(
        0.05, 0.95, eqn,
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.25)
    )

    plt.figtext(0.99, 0.02, "Student ID: 24136779", ha="right")

    plt.tight_layout()
    plt.savefig("Figure3.png", dpi=300)
    plt.show()

else:
    print("No metro journeys found in 2022 – Figure 3 not created.")

# ==========================================================
# FIGURE 4 – DISTANCE vs DURATION (ALL JOURNEYS 2022)
# ==========================================================

data_2022["Duration_minutes"] = data_2022["Duration"] * 60.0

plt.figure(figsize=(10, 6))
plt.scatter(data_2022["Distance"], data_2022["Duration_minutes"], alpha=0.6)

plt.xlabel("Distance (km)")
plt.ylabel("Duration (minutes)")
plt.title("Figure 4 – Journey Distance vs Duration (2022)")
plt.grid(True, alpha=0.3)

plt.figtext(0.99, 0.02, "Student ID: 24136779", ha="right")

plt.tight_layout()
plt.savefig("Figure4.png", dpi=300)
plt.show()
