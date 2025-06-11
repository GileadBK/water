import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Water Meter Dashboard", layout="wide")
st.markdown("**Water Meter Dashboard**")

csv_dir = 'cleaned_csv_data'
EXCLUDE_COLS = ["Year", "Month", "Week", "Day", "Time", "Basement Main", "Basement", "Date", "Hour", "CONSUMPTION (L)"]

@st.cache_data
def load_data():
    dataframes = []
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data.drop_duplicates()



def get_water_meter_columns(df):
    return [col for col in df.columns if col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[col])]

def calculate_spikes(df, meter_cols):
    # Use DateTime if available, else fallback to Date+Time or just Time
    if not meter_cols or df.empty:
        return [], []
    df = df.copy()
    if "DateTime" in df.columns:
        time_col = "DateTime"
    elif "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
        time_col = "DateTime"
    elif "Time" in df.columns:
        time_col = "Time"
    else:
        time_col = df.columns[0]  # fallback

    # Calculate total usage per row
    time_totals = df[meter_cols].sum(axis=1, numeric_only=True)
    # Use 99th percentile as spike threshold
    spike_threshold = time_totals.quantile(0.99)
    # Spikes: rows above threshold
    spikes = df.loc[time_totals > spike_threshold, time_col].dropna().tolist()
    # Inactive: rows with zero usage
    inactive = df.loc[time_totals == 0, time_col].dropna().tolist()
    return spikes, inactive



def filter_by_sidebar(combined_data):
    # --- Sidebar Filters ---
    st.sidebar.markdown("**Filters**")
    min_date = pd.to_datetime(combined_data["Date"]).min()
    max_date = pd.to_datetime(combined_data["Date"]).max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    unique_years = ["All"] + list(combined_data['Year'].unique())
    unique_months = ["All"] + list(combined_data['Month'].unique())
    unique_weeks = ["All"] + list(combined_data['Week'].unique())
    unique_days = ["All"] + list(combined_data['Day'].unique())
    unique_times = ["All"] + list(combined_data['Time'].unique())
    water_meter_options = ["All"] + list(combined_data.columns[7:16])

    water_meter = st.sidebar.multiselect("Select Water Meter", options=water_meter_options, default=["All"])
    selected_years = st.sidebar.multiselect("Select Year(s)", options=unique_years, default=["All"])
    filtered_months = unique_months if "All" in selected_years else ["All"] + list(combined_data[combined_data['Year'].isin(selected_years)]['Month'].unique())
    
    selected_months = st.sidebar.multiselect("Select Month(s)", options=filtered_months, default=["All"])
    filtered_weeks = unique_weeks if "All" in selected_years and "All" in selected_months else ["All"] + list(combined_data[
        (combined_data['Year'].isin(selected_years) if "All" not in selected_years else True) &
        (combined_data['Month'].isin(selected_months) if "All" not in selected_months else True)
    ]['Week'].unique())

    selected_weeks = st.sidebar.multiselect("Select Week(s)", options=filtered_weeks, default=["All"])
    filtered_days = unique_days if "All" in selected_years and "All" in selected_months and "All" in selected_weeks else ["All"] + list(combined_data[
        (combined_data['Year'].isin(selected_years) if "All" not in selected_years else True) &
        (combined_data['Month'].isin(selected_months) if "All" not in selected_months else True) &
        (combined_data['Week'].isin(selected_weeks) if "All" not in selected_weeks else True)
    ]['Day'].unique())

    selected_days = st.sidebar.multiselect("Select Day(s)", options=filtered_days, default=["All"])
    filtered_times = unique_times if "All" in selected_years and "All" in selected_months and "All" in selected_weeks and "All" in selected_days else ["All"] + list(combined_data[
        (combined_data['Year'].isin(selected_years) if "All" not in selected_years else True) &
        (combined_data['Month'].isin(selected_months) if "All" not in selected_months else True) &
        (combined_data['Week'].isin(selected_weeks) if "All" not in selected_weeks else True) &
        (combined_data['Day'].isin(selected_days) if "All" not in selected_days else True)
    ]['Time'].unique())
    selected_times = st.sidebar.multiselect("Select Time(s)", options=filtered_times, default=["All"])


    # --- Filtering ---
    filtered_data = combined_data.copy()
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_data = filtered_data[
            (pd.to_datetime(filtered_data["Date"]) >= start_date) &
            (pd.to_datetime(filtered_data["Date"]) <= end_date)
        ]
        if "All" not in selected_times:
            filtered_data = filtered_data[filtered_data['Time'].isin(selected_times)]
    else:
        if "All" not in selected_years:
            filtered_data = filtered_data[filtered_data['Year'].isin(selected_years)]
        if "All" not in selected_months:
            filtered_data = filtered_data[filtered_data['Month'].isin(selected_months)]
        if "All" not in selected_weeks:
            filtered_data = filtered_data[filtered_data['Week'].isin(selected_weeks)]
        if "All" not in selected_days:
            filtered_data = filtered_data[filtered_data['Day'].isin(selected_days)]
        if "All" not in selected_times:
            filtered_data = filtered_data[filtered_data['Time'].isin(selected_times)]
    if "All" not in water_meter:
        filtered_data = filtered_data[['Year', 'Month', 'Week', 'Day', 'Time', 'Date'] + water_meter]
    return filtered_data, water_meter, selected_years, selected_months, selected_weeks, selected_days, selected_times, date_range



def main():
    combined_data = load_data()
    filtered_data, water_meter, selected_years, selected_months, selected_weeks, selected_days, selected_times, date_range = filter_by_sidebar(combined_data)
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    

    # --- Metrics ---
    average_water = highest_water = total_water_flow = highest_water_type = main_water = total_incoming_vs_main = "N/A"
    if not filtered_data.empty:
        water_meter_columns = get_water_meter_columns(filtered_data)
        for col in water_meter_columns:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
        if water_meter_columns:
            meter_totals = filtered_data[water_meter_columns].sum(numeric_only=True)
            highest_water = meter_totals.max()
            highest_water_type = meter_totals.idxmax()
            total_water_flow = meter_totals.sum()
            average_water = meter_totals.mean().round(1)
        if "Basement Main" in filtered_data.columns:
            filtered_data["Basement Main"] = pd.to_numeric(filtered_data["Basement Main"], errors='coerce')
            main_water = filtered_data["Basement Main"].sum()
            total_incoming_vs_main = (f"{(100 - ((total_water_flow / main_water) * 100)).round(1)}%") if main_water > 0 else "N/A"
        if "CONSUMPTION (L)" in filtered_data.columns:
            filtered_data["CONSUMPTION (L)"] = pd.to_numeric(filtered_data["CONSUMPTION (L)"], errors='coerce')
            hanley_water = filtered_data["CONSUMPTION (L)"].sum()
            total_incoming_vs_hanley = (f"{(100 - ((hanley_water / main_water ) * 100)).round(1)}%") if hanley_water > 0 else "N/A"        
    with col1:
        st.metric("Highest Water Flow (L)", f"{highest_water:,.0f}" if isinstance(highest_water, (int, float)) and highest_water != "N/A" else highest_water, border=True)
    with col2:
        st.metric("Total Water Flow (L)", f"{total_water_flow:,.0f}" if isinstance(total_water_flow, (int, float)) and total_water_flow != "N/A" else total_water_flow, border=True)
    with col3:
        st.metric("Highest Water Type", highest_water_type, border=True)
    with col4:
        st.metric("Average Water Usage (L)", f"{average_water:,.1f}" if isinstance(average_water, (int, float)) and average_water != "N/A" else average_water, border=True)
    with col5:
        st.metric("Used/Incoming Loss %", total_incoming_vs_main, border=True)
    with col6:
        st.metric("Hanley/Incoming Loss %", total_incoming_vs_hanley, border=True)
    with col7:
        st.metric("Incoming Water Total (L)", f"{main_water:,.0f}" if isinstance(main_water, (int, float)) and main_water != "N/A" else main_water, border=True)
    with col8:
        st.metric("Hanley Water Total (L)", f"{hanley_water:,.0f}" if isinstance(hanley_water, (int, float)) and hanley_water != "N/A" else hanley_water, border=True)
    st.divider()
    col14 = st.columns(1)[0]
    with col14:
        resample_interval = st.selectbox(
            "",
            options=[("15 Minute Intervals", "15T"), ("1 Hour Intervals", "H")],
            format_func=lambda x: x[0],
            index=0
        )[1]
    if water_meter == "All" and not filtered_data.empty:
        with col14:
            with st.expander("View Water Meter Totals"):
                for meter in water_meter_columns:
                    if meter in filtered_data.columns and pd.api.types.is_numeric_dtype(filtered_data[meter]):
                        meter_total = filtered_data[meter].sum()
                        st.write(f"{meter}: {meter_total:,.0f} L")


    # --- Graph ---
    if not filtered_data.empty:
        water_meter_columns = [col for col in filtered_data.columns if col not in ["Year", "Month", "Week", "Day", "Time", "Basement Main", "Hour", "CONSUMPTION (L)"]]
        if "Date" in filtered_data.columns and "Time" in filtered_data.columns:
            filtered_data = filtered_data.copy()
            filtered_data["DateTime"] = pd.to_datetime(
                filtered_data["Date"].astype(str) + " " + filtered_data["Time"].astype(str),
                errors="coerce"
            )
            filtered_data = filtered_data.dropna(subset=["DateTime"])
            filtered_data = filtered_data.set_index("DateTime").resample(resample_interval).sum(numeric_only=True).reset_index()
            x_col = "DateTime"
        else:
            x_col = "Time" if "Time" in filtered_data.columns else filtered_data.columns[0]
        fig = go.Figure()
        for meter in water_meter_columns:
            if meter in filtered_data.columns:
                fig.add_trace(go.Scattergl(
                    x=filtered_data[x_col],
                    y=filtered_data[meter],
                    mode='lines+markers',
                    name=f"Water Meter - {meter}",
                    hovertemplate=f"<b>Water Meter:</b> {meter}<br><b>Water Flow:</b> %{{y}} L<extra></extra>"
                ))
        fig.update_layout(
            title="Water Flow Data for Selected Filters",
            xaxis_title="Date & Time",
            yaxis_title="Water Flow (L)",
            legend_title="Water Meters",
            template="plotly_white",
            xaxis=dict(
                tickformat="%Y-%m-%d<br>%H:%M",
                showgrid=True,
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(showgrid=True)
        )
        st.plotly_chart(fig, use_container_width=True)



    # --- Advanced Metrics ---
    col7, col8, col9 = st.columns(3)
    col10, col11, col12, col13 = st.columns(4)
    daytime_total = nighttime_total = 0
    day_vs_night_change = "N/A"
    delta_color_day_night = "off"
    spike_intervals_list, inactive_periods_list = [], []
    weekday_total = weekend_total = 0
    weekday_vs_weekend_change = "N/A"
    delta_color_weekday_weekend = "off"
    current_week_change = "N/A"
    delta = None
    delta_color = "off"
    safe_water_meter_columns = []

    if not filtered_data.empty:
        # Use filtered_data for all calculations
        water_meter_columns = [col for col in filtered_data.columns if col not in ["Year", "Month", "Week", "Day", "Time", "Basement Main", "Basement", "DateTime", "Hour", "CONSUMPTION (L)"]]
        for col in water_meter_columns:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')



        # --- Day vs Night Calculation ---
        if "DateTime" in filtered_data.columns:
            # Use DateTime if available (after resampling)
            filtered_data["Hour"] = filtered_data["DateTime"].dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        elif "Time" in filtered_data.columns:
            filtered_data["Hour"] = pd.to_datetime(filtered_data["Time"], errors="coerce").dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        else:
            daytime = nighttime = pd.DataFrame()

        filtered_data.drop(columns=["Hour"], inplace=True)

        daytime_total = daytime[water_meter_columns].sum().sum() if not daytime.empty else 0
        nighttime_total = nighttime[water_meter_columns].sum().sum() if not nighttime.empty else 0

        if nighttime_total > 0:
            day_vs_night_change = round(((daytime_total - nighttime_total) / (nighttime_total + daytime_total)) * 100, 1)
            delta_color_day_night = "normal" if day_vs_night_change >= 0 else "inverse"
        else:
            day_vs_night_change = "N/A"
            delta_color_day_night = "off"

        spike_intervals_list, inactive_periods_list = calculate_spikes(filtered_data, water_meter_columns)
        safe_water_meter_columns = [col for col in water_meter_columns if col in filtered_data.columns and col != "DateTime"]



        # --- Weekday vs Weekend Change with Date Range ---
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            # Filter for current period
            current_period = combined_data[
                (pd.to_datetime(combined_data["Date"]) >= start_date) &
                (pd.to_datetime(combined_data["Date"]) <= end_date)
            ]
            weekday_data = current_period[current_period["Day"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
            weekend_data = current_period[current_period["Day"].isin(["Sat", "Sun"])]
            safe_water_meter_columns = [col for col in water_meter_columns if col in weekday_data.columns and col != "DateTime"]
            weekday_total = weekday_data[safe_water_meter_columns].sum().sum() if not weekday_data.empty else 0
            weekend_total = weekend_data[safe_water_meter_columns].sum().sum() if not weekend_data.empty else 0

            # Optionally, compare to previous period (same length)
            prev_start = start_date - pd.Timedelta(days=(end_date - start_date).days + 1)
            prev_end = end_date - pd.Timedelta(days=(end_date - start_date).days + 1)
            prev_period = combined_data[
                (pd.to_datetime(combined_data["Date"]) >= prev_start) &
                (pd.to_datetime(combined_data["Date"]) <= prev_end)
            ]
            prev_weekday_data = prev_period[prev_period["Day"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
            prev_weekend_data = prev_period[prev_period["Day"].isin(["Sat", "Sun"])]
            prev_weekday_total = prev_weekday_data[safe_water_meter_columns].sum().sum() if not prev_weekday_data.empty else 0
            prev_weekend_total = prev_weekend_data[safe_water_meter_columns].sum().sum() if not prev_weekend_data.empty else 0

            # Calculate change for weekday and weekend separately (optional)
            if weekend_total > 0:
                weekday_vs_weekend_change = round(((weekday_total - weekend_total) / weekend_total) * 100, 1)
                delta_color_weekday_weekend = "normal" if weekday_vs_weekend_change >= 0 else "inverse"
            else:
                weekday_vs_weekend_change = "N/A"
                delta_color_weekday_weekend = "off"
        else:
            # --- Fallback to old logic if no date range ---
            if "Day" in combined_data.columns:
                weekday_data = combined_data[combined_data["Day"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
                weekend_data = combined_data[combined_data["Day"].isin(["Sat", "Sun"])]
                if "All" not in selected_years:
                    weekday_data = weekday_data[weekday_data["Year"].isin(selected_years)]
                    weekend_data = weekend_data[weekend_data["Year"].isin(selected_years)]
                if "All" not in selected_months:
                    weekday_data = weekday_data[weekday_data["Month"].isin(selected_months)]
                    weekend_data = weekend_data[weekend_data["Month"].isin(selected_months)]
                if "All" not in selected_weeks:
                    weekday_data = weekday_data[weekday_data["Week"].isin(selected_weeks)]
                    weekend_data = weekend_data[weekend_data["Week"].isin(selected_weeks)]
                if "All" not in selected_times:
                    weekday_data = weekday_data[weekday_data["Time"].isin(selected_times)]
                    weekend_data = weekend_data[weekend_data["Time"].isin(selected_times)]
                safe_water_meter_columns = [col for col in water_meter_columns if col in weekday_data.columns and col != "DateTime"]
                weekday_total = weekday_data[safe_water_meter_columns].sum().sum() if not weekday_data.empty else 0
                weekend_total = weekend_data[safe_water_meter_columns].sum().sum() if not weekend_data.empty else 0
                if weekend_total > 0:
                    weekday_vs_weekend_change = round(((weekday_total - weekend_total) / weekend_total) * 100, 1)
                    delta_color_weekday_weekend = "normal" if weekday_vs_weekend_change >= 0 else "inverse"
                else:
                    weekday_vs_weekend_change = "N/A"
                    delta_color_weekday_weekend = "off"



        # Week over week
        if not filtered_data.empty and date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            current_total = filtered_data[safe_water_meter_columns].sum().sum() if safe_water_meter_columns else 0
            prev_start = start_date - pd.Timedelta(days=7)
            prev_end = end_date - pd.Timedelta(days=7)
            prev_period_data = combined_data[
                (pd.to_datetime(combined_data["Date"]) >= prev_start) &
                (pd.to_datetime(combined_data["Date"]) <= prev_end)
            ]
            prev_total = prev_period_data[safe_water_meter_columns].sum().sum() if not prev_period_data.empty and safe_water_meter_columns else 0
            if prev_total > 0:
                current_week_change = round(((current_total - prev_total) / prev_total) * 100, 1)
                delta = current_week_change
                delta_color = "normal" if current_week_change >= 0 else "inverse"
        elif "All" not in selected_weeks and len(selected_weeks) == 1:
            current_week = int(selected_weeks[0])
            current_week_data = combined_data[
                (combined_data['Year'].isin(selected_years) if "All" not in selected_years else True) &
                (combined_data['Month'].isin(selected_months) if "All" not in selected_months else True) &
                (combined_data['Week'] == current_week)
            ]
            previous_week = current_week - 1
            previous_week_data = combined_data[
                (combined_data['Year'].isin(selected_years) if "All" not in selected_years else True) &
                (combined_data['Month'].isin(selected_months) if "All" not in selected_months else True) &
                (combined_data['Week'] == previous_week)
            ]
            current_week_total = current_week_data[safe_water_meter_columns].sum().sum() if not current_week_data.empty else 0
            previous_week_total = previous_week_data[safe_water_meter_columns].sum().sum() if not previous_week_data.empty else 0
            if previous_week_total > 0:
                current_week_change = round(((current_week_total - previous_week_total) / previous_week_total) * 100, 1)
                delta_color = "normal" if current_week_change >= 0 else "inverse"
                delta = current_week_change
    with col9:
        st.metric(
            label="Week Over Week Change (%)",
            value="",
            delta=f"{delta}%" if delta is not None else "None",  
            delta_color= f"{delta_color}" if delta is not None and delta < 0 else "inverse" if delta is not None else "off",  
            label_visibility="visible",
            border=True
        )

    with col10:
        with st.expander("Spike Time Intervals"):
            if spike_intervals_list:
                for interval in spike_intervals_list:
                    st.write(f"Spike at: {interval}")
            else:
                st.write("No usage spikes detected.")
    with col11:
        with st.expander("Inactive Periods"):
            if inactive_periods_list:
                for interval in inactive_periods_list:
                    st.write(f"Inactive at: {interval}")
            else:
                st.write("No inactive periods detected.")
    with col12:
        with st.expander("Day vs Night Water Usage"):
            st.write(f"Daytime Total: {daytime_total:,.0f} L")
            st.write(f"Nighttime Total: {nighttime_total:,.0f} L")
    with col7:
        st.metric(
            label="Day vs Night Change (%)",
            value="",
            delta=f"{day_vs_night_change}%" if day_vs_night_change != "N/A" else "None",
            delta_color=delta_color_day_night,
            label_visibility="visible",
            border=True
        )
    with col13:
        with st.expander("Weekday vs Weekend Water Usage"):
            st.write(f"Weekday Total: {weekday_total:,.0f} L")
            st.write(f"Weekend Total: {weekend_total:,.0f} L")
    with col8:
        st.metric(
            label="Weekday vs Weekend Change (%)",
            value="",
            delta=f"{weekday_vs_weekend_change}%" if weekday_vs_weekend_change != "N/A" else "None",
            delta_color=delta_color_weekday_weekend,
            label_visibility="visible",
            border=True
        )
    st.divider()


    print(filtered_data.head(10))
    # --- Sankey and Pie Chart ---
    if not filtered_data.empty and all(col in combined_data.columns for col in ["Year", "Month", "Week", "Day", "Time"]):
        water_meter_columns = [col for col in filtered_data.columns if col not in ["Year", "Month", "Week", "Day", "Time", "Basement Main", "Basement", "Hour", "CONSUMPTION (L)"]]
        if water_meter_columns:
            total_incoming = filtered_data["Basement Main"].sum() if "Basement Main" in filtered_data.columns else combined_data["Basement Main"].sum()
            total_water_flow = filtered_data[water_meter_columns].sum().sum()
            total_unused = total_incoming - total_water_flow
            source, target, value = [], [], []

            if total_water_flow > 0:
                source.append("Total Incoming Water")
                target.append("Total Water Flow")
                value.append(total_water_flow)

            if total_unused > 0:
                source.append("Total Incoming Water")
                target.append("Total Unused Water")
                value.append(total_unused)

            for meter in water_meter_columns:
                if meter in filtered_data.columns and pd.api.types.is_numeric_dtype(filtered_data[meter]):
                    meter_total = filtered_data[meter].sum()
                    if meter_total > 0:
                        source.append("Total Water Flow")
                        target.append(f"Meter: {meter}")
                        value.append(meter_total)

            labels = list(set(source + target))
            label_indices = {label: idx for idx, label in enumerate(labels)}
            source_indices = [label_indices[s] for s in source]
            target_indices = [label_indices[t] for t in target]

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=.5), label=labels),
                link=dict(source=source_indices, target=target_indices, value=value)
            )])
            
            fig_sankey.update_layout(title_text="Sankey Chart", font_size=10, template="plotly_white")

            pie_meter_columns = [col for col in water_meter_columns if pd.api.types.is_numeric_dtype(filtered_data[col])]
            usage_breakdown = filtered_data[pie_meter_columns].sum().reset_index()
            usage_breakdown.columns = ["Water Meter", "Total Usage"]
            fig_pie = px.pie(
                usage_breakdown,
                names="Water Meter",
                values="Total Usage",
                title="Pie Chart",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_sankey, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No water meter data available for the selected filters.")
    elif not filtered_data.empty:
        st.warning("Filtered data does not contain the required columns.")
    else:
        st.warning("No data available for the selected filters.")
    st.divider()

    # --- Total Incoming vs Hanley Incoming Line Graph ---
    if not filtered_data.empty and "Basement Main" in filtered_data.columns and "CONSUMPTION (L)" in filtered_data.columns:
        # Use DateTime if available, else fallback to Time or Date
        if "DateTime" in filtered_data.columns:
            x_col = "DateTime"
        elif "Time" in filtered_data.columns:
            x_col = "Time"
        elif "Date" in filtered_data.columns:
            x_col = "Date"
        else:
            x_col = filtered_data.columns[0]

        fig_incoming_line = go.Figure()
        fig_incoming_line.add_trace(go.Scattergl(
            x=filtered_data[x_col],
            y=filtered_data["Basement Main"],
            mode='lines+markers',
            name="Total Incoming (Basement Main)",
            line=dict(color="#1f77b4")
        ))
        fig_incoming_line.add_trace(go.Scattergl(
            x=filtered_data[x_col],
            y=filtered_data["CONSUMPTION (L)"],
            mode='lines+markers',
            name="Total Hanley (CONSUMPTION (L))",
            line=dict(color="#ff7f0e")
        ))
        fig_incoming_line.update_layout(
            title="Total Incoming vs Hanley Water Over Time",
            xaxis_title=x_col,
            yaxis_title="Liters",
            legend_title="Source",
            template="plotly_white"
        )
        st.plotly_chart(fig_incoming_line, use_container_width=True)

if __name__ == "__main__":
    main()