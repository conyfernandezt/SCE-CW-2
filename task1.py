import pandas as pd
import numpy as np

df_sku = pd.read_csv("QF_Facility_SKU_Master.csv")
df_lines = pd.read_csv("QF_Facility_Order_Lines.csv")
df_order = pd.read_csv("QF_Facility_Orders.csv")

# 1. Prepare order dates
df_order["placement_timestamp"] = pd.to_datetime(
    df_order["placement_timestamp"]
)

df_order["order_date"] = (
    df_order["placement_timestamp"].dt.normalize()
)

# 2. Attach dates to order lines
df_order_sku = df_lines.merge(
    df_order[["order_id", "order_date"]],
    on="order_id",
    how="left",
    validate="many_to_one"
)

# 3. Aggregate demand by SKU and date
daily_demand = (
    df_order_sku
    .groupby(["order_date", "sku_id"], as_index=False)["quantity"]
    .sum()
    .rename(columns={"quantity": "demand"})
)

# 4. Include every date and every SKU, including zero-demand days
dates = pd.date_range(
    df_order["order_date"].min(),
    df_order["order_date"].max(),
    freq="D"
)

# Use the master file to include all SKUs in the assortment
sku_ids = df_sku["sku_id"].dropna().unique()

full_index = pd.MultiIndex.from_product(
    [dates, sku_ids],
    names=["order_date", "sku_id"]
)

daily_demand = (
    daily_demand
    .set_index(["order_date", "sku_id"])
    .reindex(full_index, fill_value=0)
    .reset_index()
    .sort_values(["sku_id", "order_date"])
)

# 5. Product-level descriptive statistics
product_stats = (
    daily_demand.groupby("sku_id")["demand"]
    .agg(
        mean_daily="mean",
        std_daily="std",
        variance="var",
        zero_demand_share=lambda x: (x == 0).mean()
    )
)

product_stats["cv"] = (
    product_stats["std_daily"]
    / product_stats["mean_daily"].replace(0, np.nan)
)

# 6. Add calendar features
daily_demand["weekday"] = daily_demand["order_date"].dt.dayofweek
daily_demand["month"] = daily_demand["order_date"].dt.month
daily_demand["year"] = daily_demand["order_date"].dt.year

# 7. Average demand by weekday, for each SKU
df_days = (
    daily_demand.groupby(["sku_id", "weekday"])["demand"]
    .mean()
    .reset_index(name="mean_demand")
)

# 8. Average demand by calendar month, for each SKU
df_month = (
    daily_demand.groupby(["sku_id", "month"])["demand"]
    .mean()
    .reset_index(name="mean_demand")
)

# 9. Average demand by year, for each SKU
df_year = (
    daily_demand.groupby(["sku_id", "year"])["demand"]
    .mean()
    .reset_index(name="mean_demand")
)

print(product_stats.head())
print(df_days.head())
print(df_month.head())
print(df_year.head())

# Compare weekday demand against each SKU's overall mean

weekday_seasonality = df_days.merge(
    product_stats[["mean_daily"]],
    on="sku_id",
    how="left"
)

weekday_seasonality["relative_demand"] = (
    weekday_seasonality["mean_demand"]
    / weekday_seasonality["mean_daily"].replace(0, np.nan)
)

weekday_seasonality["weekday_name"] = (
    weekday_seasonality["weekday"]
    .map({
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    })
)

print(weekday_seasonality)

# Compare weekday demand against each SKU's overall mean

monthly_seasonality = df_month.merge(
    product_stats[["mean_daily"]],
    on="sku_id",
    how="left"
)

monthly_seasonality["relative_demand"] = (
    monthly_seasonality["mean_demand"]
    / monthly_seasonality["mean_daily"].replace(0, np.nan)
)

monthly_seasonality["month_name"] = (
    monthly_seasonality["month"]
    .map({
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    })
)

print(monthly_seasonality)