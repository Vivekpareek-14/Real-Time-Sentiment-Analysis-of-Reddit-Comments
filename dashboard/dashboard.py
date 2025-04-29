import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cassandra configuration
cassandra_host = os.getenv('CASSANDRA_HOST', 'cassandra')
cassandra_port = int(os.getenv('CASSANDRA_PORT', '9042'))
cassandra_keyspace = os.getenv('CASSANDRA_KEYSPACE', 'reddit_sentiment')
cassandra_username = os.getenv('CASSANDRA_USERNAME', '')
cassandra_password = os.getenv('CASSANDRA_PASSWORD', '')

# Configure page settings
st.set_page_config(
    page_title="Reddit Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 36px !important;
            font-weight: bold;
            color: #FF4500;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 24px !important;
            font-weight: bold;
            color: #1E1E1E;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .metric-container {
            background-color: #F8F8F8;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .stMetric {
            background-color: Chocolate !important;
            border-radius: 5px !important;
            padding: 10px !important;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.05) !important;
        }
        .connection-status {
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .connected {
            background-color: #4CAF50;
            color: white;
        }
        .disconnected {
            background-color: #F44336;
            color: white;
        }
        .refresh-button {
            background-color: #FF4500 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)


class CassandraConnector:
    @staticmethod
    @st.cache_resource(ttl=600)
    def init_connection():
        """Connect to Cassandra with caching for session reuse"""
        try:
            auth_provider = None
            if cassandra_username and cassandra_password:
                auth_provider = PlainTextAuthProvider(
                    username=cassandra_username,
                    password=cassandra_password
                )
            
            cluster = Cluster(
                [cassandra_host], 
                port=cassandra_port, 
                auth_provider=auth_provider
            )
            session = cluster.connect(cassandra_keyspace)
            
            # Test the connection with a simple query
            session.execute("SELECT now() FROM system.local")
            
            return cluster, session
            
        except Exception as e:
            st.session_state['connection_error'] = str(e)
            return None, None


@st.cache_data(ttl=5)  # Reduced cache time to 5 seconds
def fetch_data(_session, _refresh_flag=False):
    """Fetch data from Cassandra with caching"""
    if _session is None:
        return pd.DataFrame()
        
    try:
        # Use SimpleStatement for better query performance with pagination
        query = SimpleStatement("""
            SELECT id, created_at, subreddit, title, text, type, sentiment_label, sentiment_score
            FROM reddit_data
            LIMIT 10000
        """, fetch_size=1000)
        
        # Execute query with pagination
        data = []
        for row in _session.execute(query):
            data.append({
                'id': row.id,
                'created_at': datetime.fromtimestamp(row.created_at) if row.created_at else None,
                'subreddit': row.subreddit,
                'title': row.title or "",
                'text': row.text or "",
                'type': row.type,
                'sentiment_label': row.sentiment_label,
                'sentiment_score': float(row.sentiment_score) if row.sentiment_score is not None else 0.5
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert created_at to datetime if not already
        if 'created_at' in df.columns and df['created_at'].dtype == 'object':
            df['created_at'] = pd.to_datetime(df['created_at'])
            
        # Normalize sentiment labels to uppercase for consistency
        if 'sentiment_label' in df.columns:
            df['sentiment_label'] = df['sentiment_label'].str.upper()
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from Cassandra: {e}")
        return pd.DataFrame()


def display_metrics(df):
    """Display key metrics"""
    st.markdown('<div class="sub-header">Key Metrics</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.info("No data available to calculate metrics.")
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", len(df))
    
    with col2:
        st.metric("Unique Subreddits", df['subreddit'].nunique())
    
    with col3:
        avg_sentiment = df['sentiment_score'].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
    
    with col4:
        # Count sentiment labels
        sentiment_counts = df['sentiment_label'].value_counts()
        most_common = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"
        st.metric("Most Common Sentiment", most_common)


def display_sentiment_distribution(df):
    """Display sentiment distribution chart"""
    st.markdown('<div class="sub-header">Sentiment Distribution</div>', unsafe_allow_html=True)
    
    if df.empty or 'sentiment_label' not in df.columns:
        st.info("No sentiment data available.")
        return
        
    # Count sentiment labels
    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Check if we have data to display
    if sentiment_counts.empty:
        st.info("No sentiment distribution data available.")
        return
    
    # Create pie chart
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Distribution of Sentiment Labels',
        color='Sentiment',
        color_discrete_map={
            'POSITIVE': '#4CAF50',
            'NEGATIVE': '#F44336',
            'NEUTRAL': '#9E9E9E'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(legend_title="Sentiment", legend=dict(orientation="h", y=-0.1))
    
    st.plotly_chart(fig, use_container_width=True)


def display_sentiment_over_time(df):
    """Display sentiment over time chart with improved data handling"""
    st.markdown('<div class="sub-header">Sentiment Trends Over Time</div>', unsafe_allow_html=True)
    
    # Ensure we have data with timestamps
    if df.empty or 'created_at' not in df.columns:
        st.info("No time-based data available.")
        return
    
    try:
        # Remove rows with null timestamps
        df = df[df['created_at'].notna()].copy()
        
        # Determine appropriate time grouping based on data range
        time_range = df['created_at'].max() - df['created_at'].min()
        
        if time_range < pd.Timedelta(hours=12):
            # For data spanning less than 12 hours, group by 15-minute intervals
            df['time_group'] = df['created_at'].dt.floor('15min')
            time_title = "15-Minute Intervals"
        elif time_range < pd.Timedelta(days=1):
            # For data spanning less than 1 day, group by hour
            df['time_group'] = df['created_at'].dt.floor('H')
            time_title = "Hourly"
        elif time_range < pd.Timedelta(days=7):
            # For data spanning less than 1 week, group by day
            df['time_group'] = df['created_at'].dt.floor('D')
            time_title = "Daily"
        else:
            # For longer periods, group by week
            df['time_group'] = df['created_at'].dt.to_period('W').dt.start_time
            time_title = "Weekly"
        
        # Calculate metrics
        time_sentiment = df.groupby('time_group').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment_label': lambda x: (x == 'POSITIVE').mean()
        }).reset_index()
        
        # Flatten multi-index columns
        time_sentiment.columns = ['Time Period', 'Average Sentiment', 'Post Count', 'Positive Ratio']
        
        # Check if we have enough data points
        if len(time_sentiment) < 2:
            st.info(f"Not enough time-based data points ({len(time_sentiment)}) to display trends.")
            return
            
        # Create line chart with improved styling
        fig = go.Figure()
        
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=time_sentiment['Time Period'],
            y=time_sentiment['Average Sentiment'],
            mode='lines+markers',
            name='Avg Sentiment',
            line=dict(color='#1E88E5', width=3),
            marker=dict(size=8, color='#1E88E5'),
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>'
                        'Sentiment: %{y:.2f}<extra></extra>'
        ))
        
        # Add positive ratio line
        fig.add_trace(go.Scatter(
            x=time_sentiment['Time Period'],
            y=time_sentiment['Positive Ratio'],
            mode='lines',
            name='Positive %',
            line=dict(color='#4CAF50', width=2, dash='dot'),
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>'
                        'Positive: %{y:.1%}<extra></extra>'
        ))
        
        # Add post count bar chart on secondary y-axis
        fig.add_trace(go.Bar(
            x=time_sentiment['Time Period'],
            y=time_sentiment['Post Count'],
            name='Post Count',
            marker=dict(color='rgba(158, 158, 158, 0.3)'),
            opacity=0.5,
            yaxis='y2',
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>'
                        'Posts: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Sentiment Trend Over Time ({time_title})',
            xaxis=dict(
                title='Time',
                gridcolor='rgba(0,0,0,0.05)',
                tickformat='%Y-%m-%d %H:%M'
            ),
            yaxis=dict(
                title='Sentiment Score / Positive %',
                range=[0, 1],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis2=dict(
                title='Post Count',
                overlaying='y',
                side='right',
                gridcolor='rgba(0,0,0,0)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        with st.expander("Time Period Summary"):
            st.write(f"Time range: {df['created_at'].min()} to {df['created_at'].max()}")
            st.write(f"Time grouping: {time_title}")
            st.write(f"Data points: {len(time_sentiment)}")
            
    except Exception as e:
        st.error(f"Error creating sentiment over time chart: {str(e)}")


def display_sentiment_by_subreddit(df):
    """Display sentiment by subreddit chart"""
    st.markdown('<div class="sub-header">Sentiment by Subreddit</div>', unsafe_allow_html=True)
    
    if df.empty or 'subreddit' not in df.columns or df['subreddit'].nunique() == 0:
        st.info("No subreddit data available.")
        return
        
    try:
        # Group by subreddit and calculate average sentiment
        subreddit_sentiment = df.groupby('subreddit').agg({
            'sentiment_score': 'mean',
            'id': 'count'
        }).reset_index()
        subreddit_sentiment.columns = ['Subreddit', 'Average Sentiment', 'Post Count']
        
        # Sort by post count
        subreddit_sentiment = subreddit_sentiment.sort_values('Post Count', ascending=False)
        
        # Take top 10 subreddits by post count
        top_subreddits = subreddit_sentiment.head(10)
        
        # Check if we have data to display
        if top_subreddits.empty:
            st.info("Not enough subreddit data to display trends.")
            return
            
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=top_subreddits['Subreddit'],
            x=top_subreddits['Average Sentiment'],
            orientation='h',
            marker=dict(
                color=top_subreddits['Average Sentiment'],
                colorscale='RdYlGn',  # Red to Yellow to Green
                cmin=0,
                cmax=1,
                colorbar=dict(title='Sentiment Score')
            ),
            text=top_subreddits['Post Count'].apply(lambda x: f"{x} posts"),
            textposition='auto',
            name='Average Sentiment'
        ))
        
        # Update layout
        fig.update_layout(
            title='Average Sentiment by Subreddit (Top 10 by Post Count)',
            xaxis=dict(title='Average Sentiment Score', range=[0, 1]),
            yaxis=dict(title='Subreddit'),
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating sentiment by subreddit chart: {e}")


def display_recent_posts(df):
    """Display recent posts with sentiment"""
    st.markdown('<div class="sub-header">Recent Posts with Sentiment Analysis</div>', unsafe_allow_html=True)
    
    if df.empty or 'created_at' not in df.columns:
        st.info("No recent posts available.")
        return
        
    try:
        # Sort by created_at and take most recent posts
        recent_df = df.sort_values('created_at', ascending=False).head(10)
        
        # Display as expandable cards
        for _, row in recent_df.iterrows():
            title = row['title'][:80] + "..." if len(row['title']) > 80 else row['title']
            with st.expander(f"{title} (r/{row['subreddit']})"):
                sentiment_label = row['sentiment_label'].upper() if pd.notna(row['sentiment_label']) else "UNKNOWN"
                sentiment_color = "green" if sentiment_label == 'POSITIVE' else "red" if sentiment_label == 'NEGATIVE' else "gray"
                
                post_text = row['text'] if isinstance(row['text'], str) else ""
                post_text = post_text[:500] + "..." if len(post_text) > 500 else post_text
                
                st.markdown(f"""
                    **Subreddit**: r/{row['subreddit']}  
                    **Posted**: {row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['created_at']) else 'Unknown'}  
                    **Sentiment**: <span style="color:{sentiment_color};">{sentiment_label} ({row['sentiment_score']:.2f})</span>  
                    **Content**: {post_text}
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying recent posts: {e}")


def display_connection_status(connected):
    """Display connection status"""
    if connected:
        st.markdown(
            '<div class="connection-status connected">✅ Connected to Cassandra</div>',
            unsafe_allow_html=True
        )
    else:
        error_msg = st.session_state.get('connection_error', 'Unknown error')
        st.markdown(
            f'<div class="connection-status disconnected">❌ Not connected to Cassandra: {error_msg}</div>',
            unsafe_allow_html=True
        )


def main():
    """Main function for Streamlit dashboard"""
    # Initialize session state
    if 'connection_error' not in st.session_state:
        st.session_state['connection_error'] = None
    if 'refresh_flag' not in st.session_state:
        st.session_state['refresh_flag'] = False
    
    # Display header
    st.markdown('<div class="main-header">Reddit Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Add sidebar with refresh options
    with st.sidebar:
        st.title("Dashboard Controls")
        
        # Add refresh button with better visual feedback
        if st.button("Refresh Data", key="refresh_button", help="Force refresh all data"):
            st.session_state['refresh_flag'] = not st.session_state['refresh_flag']
            st.toast("Refreshing data...")
            time.sleep(0.5)
            st.rerun()
        
        # Connection reset button
        if st.button("Reset Connection", key="reset_connection"):
            st.cache_resource.clear()
            st.session_state['connection_error'] = None
            st.toast("Resetting connection...")
            time.sleep(1)
            st.rerun()
        
        # Add auto-refresh options
        auto_refresh = st.selectbox(
            "Auto-refresh interval",
            ["Off", "10 seconds", "30 seconds", "1 minute"],
            index=0
        )
        
        # Debug section
        with st.expander("Debug Tools"):
            if st.button("Clear All Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.toast("Cleared all cache!", icon="🧹")
                time.sleep(1)
                st.rerun()
            
            if st.button("Show Raw Data Sample"):
                st.session_state['show_raw_data'] = not st.session_state.get('show_raw_data', False)
    
    # Connect to Cassandra
    cluster, session = CassandraConnector.init_connection()
    
    # Display connection status
    display_connection_status(session is not None)
    
    if not session:
        # Display error and provide options to retry
        st.error("Could not connect to Cassandra database. Please check your configuration.")
        
        if st.button("Retry Connection", key="retry_connection"):
            st.cache_resource.clear()
            st.session_state['connection_error'] = None
            st.rerun()
            
        # Show debugging info in expander
        with st.expander("Connection Details"):
            st.code(f"""
            Host: {cassandra_host}
            Port: {cassandra_port}
            Keyspace: {cassandra_keyspace}
            Username: {"Set" if cassandra_username else "Not Set"}
            Password: {"Set" if cassandra_password else "Not Set"}
            Error: {st.session_state.get('connection_error', 'Unknown error')}
            """)
            
        # Add help text for common errors
        st.markdown("""
        ### Common Solutions:
        1. Check that the Cassandra service is running
        2. Verify the keyspace exists and table schema is correct
        3. Check network connectivity and firewall settings
        4. Ensure credentials are correct if authentication is enabled
        """)
        
        # Handle auto-refresh even when connection fails
        if auto_refresh != "Off":
            refresh_times = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60}
            refresh_time = refresh_times[auto_refresh]
            
            st.sidebar.text(f"Will retry connection in {refresh_time} seconds")
            time.sleep(refresh_time)
            st.rerun()
            
        return
    
    # Connection successful, fetch data
    with st.spinner("Loading data..."):
        df = fetch_data(session, st.session_state['refresh_flag'])
    
    # Debug raw data display
    if st.session_state.get('show_raw_data', False):
        with st.expander("Raw Data Sample"):
            st.write(f"Data last fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Total records: {len(df)}")
            if not df.empty:
                st.write(f"Newest post: {df['created_at'].max().strftime('%Y-%m-%d %H:%M:%S')}")
            st.dataframe(df.head(3))
    
    if df.empty:
        st.warning("No data available in the database. Please ensure the Reddit producer and Spark consumer are running.")
        
        # Provide schema check functionality
        if st.button("Check Table Schema"):
            try:
                rows = session.execute(f"SELECT column_name, type FROM system_schema.columns WHERE keyspace_name = '{cassandra_keyspace}' AND table_name = 'reddit_data'")
                st.success("Table schema retrieved successfully!")
                
                schema_data = [(r.column_name, r.type) for r in rows]
                schema_df = pd.DataFrame(schema_data, columns=["Column", "Type"])
                st.dataframe(schema_df)
                
                # Check for specific columns needed
                needed_columns = ["id", "created_at", "subreddit", "title", "text", "type", "sentiment_label", "sentiment_score"]
                existing_columns = [r[0] for r in schema_data]
                
                missing = [col for col in needed_columns if col not in existing_columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.success("All required columns are present!")
                    
            except Exception as e:
                st.error(f"Error checking schema: {e}")
        
        # Handle auto-refresh even when no data is available
        if auto_refresh != "Off":
            refresh_times = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60}
            refresh_time = refresh_times[auto_refresh]
            
            st.sidebar.text(f"Auto-refreshing in {refresh_time} seconds")
            time.sleep(refresh_time)
            st.rerun()
            
        return
    
    # Display dashboard components
    display_metrics(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_sentiment_distribution(df)
    
    with col2:
        display_sentiment_by_subreddit(df)
    
    display_sentiment_over_time(df)
    display_recent_posts(df)
    
    # Handle auto-refresh
    if auto_refresh != "Off":
        refresh_times = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60}
        refresh_time = refresh_times[auto_refresh]
        
        time.sleep(refresh_time)
        st.rerun()


if __name__ == "__main__":
    main()