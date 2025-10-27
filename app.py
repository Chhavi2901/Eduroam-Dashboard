import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from log_generator import extract_summary_tables_from_stream, parse_and_generate_excel

st.set_page_config(page_title="Eduroam Dashboard", layout="wide")
st.title("📡 Eduroam Log Dashboard")

# === File selection ===
st.sidebar.header("🗂 Select Log File (.log or .txt)")
log_path = st.text_input("Enter full path to your log file (e.g., C:/eduroam_logs/old_logs.log)", key="log_path_input")

if not (log_path and os.path.exists(log_path)):
    st.info("Please enter a valid file path to start.")
    st.stop()

# === Stream parsing ===
file_size_gb = os.path.getsize(log_path) / (1024 ** 3)
with st.spinner(f"⏳ Parsing {file_size_gb:.2f} GB log file in chunks..."):
    with open(log_path, "rb") as f:
        df_access, df_fticks, access_csv, fticks_csv = extract_summary_tables_from_stream(f)

st.success(f"✅ Parsed {len(df_access):,} access log samples and {len(df_fticks):,} F-TICKS samples.")
st.caption(f"Full parsed data saved to disk:\n• {access_csv}\n• {fticks_csv}")

if df_access.empty and df_fticks.empty:
    st.error("No valid data found in this log file.")
    st.stop()

# === Optional sampling for visualization ===
MAX_ROWS = 100_000  # Reduced for faster processing while maintaining quality
if len(df_fticks) > MAX_ROWS:
    # Use stratified sampling to maintain data distribution
    if 'RESULT' in df_fticks.columns:
        df_fticks_vis = df_fticks.groupby('RESULT', group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_ROWS // df_fticks['RESULT'].nunique()), random_state=1)
        ).reset_index(drop=True)
    else:
        df_fticks_vis = df_fticks.sample(n=MAX_ROWS, random_state=1)
else:
    df_fticks_vis = df_fticks

if len(df_access) > MAX_ROWS:
    # Use stratified sampling to maintain data distribution  
    if 'Event' in df_access.columns:
        df_access_vis = df_access.groupby('Event', group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_ROWS // df_access['Event'].nunique()), random_state=1)
        ).reset_index(drop=True)
    else:
        df_access_vis = df_access.sample(n=MAX_ROWS, random_state=1)
else:
    df_access_vis = df_access

# === Derived fields ===
@st.cache_data
def extract_domain_vectorized(usernames):
    """Optimized domain extraction using vectorized operations"""
    if usernames.empty:
        return pd.Series(dtype='object')
    return usernames.str.extract(r'@(.+)$')[0].str.strip().str.lower()

df_access["Domain"] = extract_domain_vectorized(df_access["Username"] if not df_access.empty else pd.Series())

# =====================================================
# ACCESS LOGS DASHBOARD
# =====================================================
st.sidebar.header("🔍 Filter Access Logs")
access_df_filtered = df_access_vis.copy()

# Filters
min_date, max_date = df_access["Timestamp"].min(), df_access["Timestamp"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], key="date_range_filter")
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    access_df_filtered = access_df_filtered[
        (access_df_filtered["Timestamp"] >= start) & (access_df_filtered["Timestamp"] <= end)
    ]

selected_events = st.sidebar.multiselect(
    "Access Event Types",
    access_df_filtered["Event"].dropna().unique(),
    default=access_df_filtered["Event"].dropna().unique(),
    key="events_filter"
)
access_df_filtered = access_df_filtered[access_df_filtered["Event"].isin(selected_events)]

selected_domains = st.sidebar.multiselect(
    "Filter by Domain", sorted(access_df_filtered["Domain"].dropna().unique()), key="domains_filter"
)
if selected_domains:
    access_df_filtered = access_df_filtered[access_df_filtered["Domain"].isin(selected_domains)]

st.subheader("🔐 Access Logs")
st.dataframe(access_df_filtered.drop(columns=["Domain"]).reset_index(drop=True), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Event Type Distribution")
    st.bar_chart(access_df_filtered["Event"].value_counts())
with col2:
    st.markdown("#### Top Domains")
    st.bar_chart(access_df_filtered["Domain"].value_counts().head(5))

st.markdown("#### 🍩 Top 10 Domains")
top_domains = access_df_filtered["Domain"].value_counts().head(10).reset_index()
top_domains.columns = ["Domain", "Count"]
fig = px.pie(top_domains, names="Domain", values="Count", title="Top 10 Domains")
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# F-TICKS DASHBOARD
# =====================================================
st.sidebar.header("🔍 Filter F-TICKS Logs")
fticks_df_filtered = df_fticks_vis.copy()

st.sidebar.markdown("---")
st.sidebar.header("🌏 Roaming Events")
roaming_type = st.sidebar.selectbox(
    "Select Roaming Type",
    ["All", "Indian to Foreign", "Foreign to Indian", "Local (Indian to Indian)"],
    key="roaming_type_selector"
)

def is_foreign_realm(r):
    return pd.notnull(r) and not str(r).lower().endswith(".in")

def is_indian_realm(r):
    return pd.notnull(r) and str(r).lower().endswith(".in")

if roaming_type == "Indian to Foreign":
    fticks_df_filtered = fticks_df_filtered[
        (fticks_df_filtered["VISCOUNTRY"].str.upper() != "IN") &
        (fticks_df_filtered["RESULT"].notnull())
    ]
elif roaming_type == "Foreign to Indian":
    fticks_df_filtered = fticks_df_filtered[
        (fticks_df_filtered["VISCOUNTRY"].str.upper() == "IN") &
        (fticks_df_filtered["RESULT"].notnull()) &
        (fticks_df_filtered["REALM"].apply(is_foreign_realm))
    ]
elif roaming_type == "Local (Indian to Indian)":
    fticks_df_filtered = fticks_df_filtered[
        (fticks_df_filtered["VISCOUNTRY"].str.upper() == "IN") &
        (fticks_df_filtered["RESULT"].notnull()) &
        (fticks_df_filtered["REALM"].apply(is_indian_realm))
    ]

# === Roaming event summary ===
st.markdown("---")

def is_foreign_realm(realm):
    return pd.notnull(realm) and not str(realm).strip().lower().endswith(".in")

def is_indian_realm(realm):
    return pd.notnull(realm) and str(realm).strip().lower().endswith(".in")

# --- FILTER DATASET BASED ON ROAMING TYPE ---
roaming_df = df_fticks_vis.copy()

# === MERGE ACCESS AND F-TICKS DATA FOR RICHER ROAMING CONTEXT ===
def merge_access_fticks_data(fticks_df, access_df):
    """
    Optimized merge function for F-TICKS roaming data with Access logs.
    Uses vectorized operations and efficient indexing for faster processing.
    """
    if access_df.empty or fticks_df.empty:
        return fticks_df
    
    # Create a copy to avoid modifying original
    merged_df = fticks_df.copy()
    
    # Pre-process access data for faster lookups
    access_enriched = access_df.copy()
    
    # Vectorized string operations
    access_enriched['Username_Lower'] = access_enriched['Username'].str.lower().str.strip()
    access_enriched['Domain'] = access_enriched['Username'].str.extract(r'@(.+)$')[0]
    
    # Create time-based index for faster lookups
    access_enriched = access_enriched.sort_values('Timestamp').reset_index(drop=True)
    
    # Initialize new columns efficiently
    new_columns = ['Event', 'Username', 'Source', 'Destination', 'ServerIP', 'MAC', 'Access_Match_Type']
    for col in new_columns:
        merged_df[col] = None
    
    # Time window for matching (5 minutes)
    time_window = pd.Timedelta(minutes=5)
    
    # Vectorized operations for realm processing
    merged_df['realm_lower'] = merged_df['REALM'].str.lower().str.strip()
    merged_df['realm_username'] = merged_df['REALM'].str.extract(r'^([^@]+)@')[0].str.lower()
    merged_df['realm_domain'] = merged_df['REALM'].str.extract(r'@(.+)$')[0].str.lower()
    
    # Process in batches for memory efficiency
    batch_size = 1000
    total_batches = len(merged_df) // batch_size + (1 if len(merged_df) % batch_size else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(merged_df))
        batch_df = merged_df.iloc[start_idx:end_idx].copy()
        
        for idx in range(len(batch_df)):
            actual_idx = start_idx + idx
            row = batch_df.iloc[idx]
            
            fticks_time = row['Timestamp']
            if pd.isna(fticks_time):
                continue
            
            # Fast time window filtering using binary search-like approach
            time_mask = (
                (access_enriched['Timestamp'] >= fticks_time - time_window) &
                (access_enriched['Timestamp'] <= fticks_time + time_window)
            )
            time_filtered = access_enriched[time_mask]
            
            if time_filtered.empty:
                continue
            
            matched_access = None
            match_type = None
            
            # Strategy 1: Exact REALM match (fastest)
            if pd.notnull(row['realm_lower']) and '@' in str(row['REALM']):
                exact_match = time_filtered[time_filtered['Username_Lower'] == row['realm_lower']]
                if not exact_match.empty:
                    matched_access = exact_match.iloc[0]
                    match_type = "Exact_REALM_Match"
            
            # Strategy 2: Username part match
            if matched_access is None and pd.notnull(row['realm_username']):
                username_match = time_filtered[time_filtered['Username_Lower'].str.startswith(row['realm_username'], na=False)]
                if not username_match.empty:
                    matched_access = username_match.iloc[0]
                    match_type = "Username_Part_Match"
            
            # Strategy 3: Domain-based matching
            if matched_access is None and pd.notnull(row['realm_domain']):
                domain_match = time_filtered[time_filtered['Domain'].str.lower() == row['realm_domain']]
                if not domain_match.empty:
                    # Get closest time match
                    domain_match = domain_match.copy()
                    domain_match['time_diff'] = abs(domain_match['Timestamp'] - fticks_time)
                    matched_access = domain_match.loc[domain_match['time_diff'].idxmin()]
                    match_type = "Domain_Match"
            
            # Strategy 4: Closest time match (fallback)
            if matched_access is None:
                time_filtered_copy = time_filtered.copy()
                time_filtered_copy['time_diff'] = abs(time_filtered_copy['Timestamp'] - fticks_time)
                matched_access = time_filtered_copy.loc[time_filtered_copy['time_diff'].idxmin()]
                match_type = "Time_Proximity_Match"
            
            # Apply matched data efficiently
            if matched_access is not None:
                merged_df.at[actual_idx, 'Event'] = matched_access['Event']
                merged_df.at[actual_idx, 'Username'] = matched_access['Username']
                merged_df.at[actual_idx, 'Source'] = matched_access['Source']
                merged_df.at[actual_idx, 'Destination'] = matched_access['Destination']
                merged_df.at[actual_idx, 'ServerIP'] = matched_access['ServerIP']
                merged_df.at[actual_idx, 'Access_Match_Type'] = match_type
                
                # Fast MAC detection
                source_str = str(matched_access['Source'])
                if len(source_str) == 17 and ':' in source_str or '-' in source_str:
                    if re.match(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$', source_str):
                        merged_df.at[actual_idx, 'MAC'] = source_str
    
    # Clean up temporary columns
    merged_df = merged_df.drop(columns=['realm_lower', 'realm_username', 'realm_domain'], errors='ignore')
    
    return merged_df

# Apply the merge function
with st.spinner("🔗 Merging Access and F-TICKS data for enhanced roaming context..."):
    roaming_df = merge_access_fticks_data(roaming_df, df_access_vis)

if roaming_type == "Indian to Foreign":

    # Correct logic: REALM is Indian (.in), VISCOUNTRY is not IN
    roaming_df = roaming_df[
        (roaming_df["VISCOUNTRY"].str.upper() != "IN") &
        (roaming_df["REALM"].apply(is_indian_realm))
    ]

    home_insts = sorted(roaming_df["REALM"].dropna().unique())
    visit_insts = sorted(roaming_df["VISINST"].dropna().unique())

    selected_home = st.sidebar.selectbox("🏠 Home Institution (Indian Realm)", ["All"] + home_insts, key="home_indian_to_foreign")
    selected_visit = st.sidebar.selectbox("🌍 Visiting Institution (Foreign)", ["All"] + visit_insts, key="visit_foreign")

    if selected_home != "All":
        roaming_df = roaming_df[roaming_df["REALM"] == selected_home]
    if selected_visit != "All":
        roaming_df = roaming_df[roaming_df["VISINST"] == selected_visit]

elif roaming_type == "Foreign to Indian":
    roaming_df = roaming_df[
        (roaming_df["VISCOUNTRY"].str.upper() == "IN") &
        (roaming_df["REALM"].apply(is_foreign_realm))
    ]

    home_insts = sorted(roaming_df["REALM"].dropna().unique())
    visit_insts = sorted(roaming_df["VISINST"].dropna().unique())

    selected_home = st.sidebar.selectbox("🏠 Home Institution (Foreign Realm)", ["All"] + home_insts, key="home_foreign_to_indian")
    selected_visit = st.sidebar.selectbox("🌏 Visiting Institution (Indian)", ["All"] + visit_insts, key="visit_indian")

    if selected_home != "All":
        roaming_df = roaming_df[roaming_df["REALM"] == selected_home]
    if selected_visit != "All":
        roaming_df = roaming_df[roaming_df["VISINST"] == selected_visit]

elif roaming_type == "Local (Indian to Indian)":
    roaming_df = roaming_df[
        (roaming_df["VISCOUNTRY"].str.upper() == "IN") &
        (roaming_df["REALM"].apply(is_indian_realm))
    ]
    home_insts = sorted(roaming_df["REALM"].dropna().unique())
    visit_insts = sorted(roaming_df["VISINST"].dropna().unique())

    selected_home = st.sidebar.selectbox("🏠 Home Institution (Indian Realm)", ["All"] + home_insts, key="home_local_indian")
    selected_visit = st.sidebar.selectbox("🏫 Visiting Institution (Indian)", ["All"] + visit_insts, key="visit_local_indian")

    if selected_home != "All":
        roaming_df = roaming_df[roaming_df["REALM"] == selected_home]
    if selected_visit != "All":
        roaming_df = roaming_df[roaming_df["VISINST"] == selected_visit]
else:
    # "All" selected → no filtering
    selected_home = "All"
    selected_visit = "All"

# --- DISPLAY FILTERED ROAMING TABLE ---
if not roaming_df.empty:
    st.markdown("### ✈️ Enhanced Roaming Log Details")
    
    # Show merge statistics
    merge_stats = roaming_df['Access_Match_Type'].value_counts()
    if not merge_stats.empty:
        st.markdown("#### 🔗 Data Merge Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        total_records = len(roaming_df)
        merged_records = roaming_df['Access_Match_Type'].notna().sum()
        
        with col_stat1:
            st.metric("Total Roaming Records", total_records)
        with col_stat2:
            st.metric("Records with Access Data", merged_records)
        with col_stat3:
            st.metric("Merge Success Rate", f"{(merged_records/total_records*100):.1f}%")
        with col_stat4:
            if not merge_stats.empty:
                best_match_type = merge_stats.index[0]
                st.metric("Primary Match Type", best_match_type.replace("_", " "))
        
        # Show detailed merge statistics
        if st.expander("📊 View Detailed Merge Statistics"):
            merge_stats_df = merge_stats.reset_index()
            merge_stats_df.columns = ['Match Type', 'Count']
            merge_stats_df['Percentage'] = (merge_stats_df['Count'] / total_records * 100).round(1)
            st.dataframe(merge_stats_df, use_container_width=True)

    roaming_df_display = roaming_df.copy().reset_index(drop=True)
    roaming_df_display.index += 1  # start S.No at 1

    # Create richer display table
    roaming_df_display["Home Country"] = roaming_df_display["REALM"].apply(
        lambda r: "IN" if str(r).lower().endswith(".in") else "Foreign"
    )
    roaming_df_display["Home University"] = roaming_df_display["REALM"]
    roaming_df_display["Visiting Country"] = roaming_df_display["VISCOUNTRY"]
    roaming_df_display["Visiting University"] = roaming_df_display["VISINST"]

    # Enhanced column ordering with merged data
    columns_order = [
        "Timestamp", "Username", "Event", "RESULT", 
        "Source", "Destination", "ServerIP", "MAC",
        "Home Country", "Home University",
        "Visiting Country", "Visiting University",
        "CSI", "Access_Match_Type"
    ]

    # Keep only existing columns
    columns_order = [c for c in columns_order if c in roaming_df_display.columns]
    
    # Add filtering options for the enhanced table
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        show_only_merged = st.checkbox("Show only records with Access data", key="show_merged_only")
    
    with col_filter2:
        if 'RESULT' in roaming_df_display.columns:
            result_filter = st.selectbox(
                "Filter by Auth Result", 
                ["All"] + sorted(roaming_df_display['RESULT'].dropna().unique().tolist()),
                key="result_filter"
            )
    
    with col_filter3:
        if 'Event' in roaming_df_display.columns:
            event_filter = st.selectbox(
                "Filter by Access Event", 
                ["All"] + sorted(roaming_df_display['Event'].dropna().unique().tolist()),
                key="event_filter"
            )
    
    # Apply additional filters
    filtered_display = roaming_df_display.copy()
    
    if show_only_merged:
        filtered_display = filtered_display[filtered_display['Access_Match_Type'].notna()]
    
    if 'result_filter' in locals() and result_filter != "All":
        filtered_display = filtered_display[filtered_display['RESULT'] == result_filter]
    
    if 'event_filter' in locals() and event_filter != "All":
        filtered_display = filtered_display[filtered_display['Event'] == event_filter]

    st.dataframe(
        filtered_display[columns_order],
        use_container_width=True,
        hide_index=False,
    )
    
    # Enhanced export options
    st.markdown("#### 📤 Export Enhanced Roaming Data")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_roaming = filtered_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Enhanced Roaming Data (CSV)", 
            data=csv_roaming, 
            file_name="enhanced_roaming_data.csv", 
            mime="text/csv",
            key="download_enhanced_roaming"
        )
    
    with col_exp2:
        # Summary stats
        summary_stats = {
            'Total Records': len(filtered_display),
            'Unique Users': filtered_display['Username'].nunique() if 'Username' in filtered_display.columns else 0,
            'Unique Home Institutions': filtered_display['Home University'].nunique(),
            'Unique Visiting Institutions': filtered_display['Visiting University'].nunique(),
            'Success Rate': f"{(filtered_display['RESULT'].eq('OK').sum() / len(filtered_display) * 100):.1f}%" if 'RESULT' in filtered_display.columns else "N/A"
        }
        
        if st.button("📊 Generate Summary Report", key="summary_report_btn"):
            st.json(summary_stats)

    # === Enhanced Roaming Analytics ===
    if len(filtered_display) > 0:
        st.markdown("#### 📈 Enhanced Roaming Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'Access_Match_Type' in filtered_display.columns:
                st.markdown("##### Data Integration Success")
                match_counts = filtered_display['Access_Match_Type'].value_counts()
                if not match_counts.empty:
                    # Optimize chart creation
                    fig_match = px.pie(
                        values=match_counts.values, 
                        names=[name.replace('_', ' ') for name in match_counts.index],
                        title="Access-FTicks Merge Methods"
                    )
                    fig_match.update_layout(height=400)  # Fixed height for better performance
                    st.plotly_chart(fig_match, use_container_width=True)
        
        with col_chart2:
            if 'Event' in filtered_display.columns and filtered_display['Event'].notna().any():
                st.markdown("##### Access Event Distribution")
                event_counts = filtered_display['Event'].value_counts().head(10)  # Limit to top 10
                if not event_counts.empty:
                    fig_events = px.bar(
                        x=event_counts.values,
                        y=event_counts.index,
                        orientation='h',
                        title="Authentication Events",
                        labels={'x': 'Count', 'y': 'Event Type'}
                    )
                    fig_events.update_layout(height=400)  # Fixed height for better performance
                    st.plotly_chart(fig_events, use_container_width=True)
        
        # Roaming flow analysis (optimized)
        if 'Home Country' in filtered_display.columns and 'Visiting Country' in filtered_display.columns:
            st.markdown("##### 🌍 Roaming Flow Analysis")
            
            # Create flow summary with optimized aggregation
            flow_summary = (filtered_display
                          .groupby(['Home Country', 'Visiting Country'], observed=True)
                          .size()
                          .reset_index(name='Count')
                          .sort_values('Count', ascending=False)
                          .head(10))  # Limit to top 10 flows
            
            flow_summary['Flow'] = flow_summary['Home Country'] + ' → ' + flow_summary['Visiting Country']
            
            if len(flow_summary) > 0:
                col_flow1, col_flow2 = st.columns(2)
                
                with col_flow1:
                    fig_flow = px.bar(
                        flow_summary,
                        x='Count',
                        y='Flow',
                        orientation='h',
                        title="Top 10 Roaming Flows",
                        labels={'Count': 'Number of Sessions', 'Flow': 'Country Flow'}
                    )
                    fig_flow.update_layout(height=400)
                    st.plotly_chart(fig_flow, use_container_width=True)
                
                with col_flow2:
                    # Time-based analysis (optimized with sampling if needed)
                    if 'Timestamp' in filtered_display.columns and len(filtered_display) > 1:
                        time_sample = filtered_display.sample(min(10000, len(filtered_display)), random_state=1)
                        time_sample = time_sample.copy()
                        time_sample['Hour'] = time_sample['Timestamp'].dt.hour
                        hourly_roaming = time_sample['Hour'].value_counts().sort_index()
                        
                        fig_time = px.line(
                            x=hourly_roaming.index,
                            y=hourly_roaming.values,
                            title="Roaming Activity by Hour of Day",
                            labels={'x': 'Hour of Day', 'y': 'Number of Sessions'}
                        )
                        fig_time.update_layout(height=400)
                        st.plotly_chart(fig_time, use_container_width=True)

else:
    st.warning("No roaming records found for selected filters.")


# === F-TICKS charts ===
st.markdown("---")
st.subheader("🔵 F-TICKS Overview")
st.dataframe(fticks_df_filtered.reset_index(drop=True), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("#### Auth Result Distribution")
    st.bar_chart(fticks_df_filtered["RESULT"].value_counts())
with col4:
    st.markdown("#### Top VISINST")
    st.bar_chart(fticks_df_filtered["VISINST"].value_counts().head(5))

# =====================================================
# EXPORT SECTION
# =====================================================
st.markdown("---")
st.subheader("📤 Export Reports")

col5, col6 = st.columns(2)
with col5:
    csv_access = access_df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Access Logs (CSV)", data=csv_access, file_name="access_filtered.csv", mime="text/csv")

with col6:
    csv_fticks = fticks_df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download F-TICKS Logs (CSV)", data=csv_fticks, file_name="fticks_filtered.csv", mime="text/csv")

if st.button("Generate Excel Report with Charts", key="excel_report_button"):
    with st.spinner("📊 Generating Excel..."):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        excel_file = parse_and_generate_excel(raw_text)
        st.download_button(
            "⬇️ Download Eduroam Excel File",
            data=excel_file,
            file_name="Eduroam_Log_Analyzer.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
