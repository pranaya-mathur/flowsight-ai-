"""
FlowSight AI - Streamlit Dashboard

Interactive dashboard for shipment delay predictions with
enrichment visualization and LLM explanations.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="FlowSight AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def make_prediction(shipment_data, apply_enrichment=True, explain=True):
    """Call prediction API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=shipment_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"


def plot_probability_gauge(probability, title="Delay Probability"):
    """Create a gauge chart for probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_enrichment_waterfall(raw_prob, vendor_adj, final_prob):
    """Create waterfall chart showing enrichment impact."""
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Base Model", "Vendor Adjustment", "Final Prediction"],
        y=[raw_prob * 100, vendor_adj * 100, final_prob * 100],
        text=[f"{raw_prob*100:.1f}%", f"{vendor_adj*100:+.1f}%", f"{final_prob*100:.1f}%"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#28a745"}},
        increasing={"marker": {"color": "#dc3545"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))
    fig.update_layout(
        title="Enrichment Impact Analysis",
        showlegend=False,
        height=400,
        yaxis_title="Delay Probability (%)"
    )
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">📦 FlowSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Supply Chain Delay Prediction</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API Server is not running! Start it with: `python -m uvicorn src.api.main:app --reload`")
        st.stop()
    
    st.success("✅ Connected to FlowSight API")
    
    # Sidebar - Input Form
    st.sidebar.header("📋 Shipment Details")
    
    with st.sidebar.form("shipment_form"):
        st.subheader("Route Information")
        origin_city = st.text_input("Origin City", "Mumbai")
        destination_city = st.text_input("Destination City", "Delhi")
        origin_state = st.text_input("Origin State", "Maharashtra")
        destination_state = st.text_input("Destination State", "Delhi")
        
        st.subheader("Product Details")
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Apparel", "FMCG", "Automotive", "Pharmaceuticals"]
        )
        quantity = st.number_input("Quantity", min_value=1, value=100)
        weight_kg = st.number_input("Weight (kg)", min_value=0.1, value=500.0)
        value_inr = st.number_input("Value (₹)", min_value=1.0, value=50000.0)
        
        st.subheader("Logistics Details")
        supplier_name = st.text_input("Vendor ID", "V001")
        carrier_name = st.selectbox(
            "Carrier",
            ["BlueDart", "DTDC", "Delhivery", "Ecom Express", "FedEx"]
        )
        truck_type = st.selectbox("Truck Type", ["LCV", "MHCV", "Trailer"])
        month = st.selectbox(
            "Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
        
        st.subheader("Options")
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.6, 0.1)
        apply_enrichment = st.checkbox("Apply Enrichment", value=True)
        explain = st.checkbox("Generate Explanation", value=True)
        
        submitted = st.form_submit_button("🔮 Predict Delay", use_container_width=True)
    
    # Main area
    if submitted:
        # Prepare payload
        payload = {
            "origin_city": origin_city,
            "destination_city": destination_city,
            "origin_state": origin_state,
            "destination_state": destination_state,
            "product_category": product_category,
            "supplier_name": supplier_name,
            "carrier_name": carrier_name,
            "truck_type": truck_type,
            "month": month,
            "quantity": quantity,
            "weight_kg": weight_kg,
            "value_inr": value_inr,
            "risk_score": risk_score,
            "apply_enrichment": apply_enrichment,
            "explain": explain
        }
        
        # Make prediction
        with st.spinner("🔄 Analyzing shipment..."):
            result, error = make_prediction(payload, apply_enrichment, explain)
        
        if error:
            st.error(f"❌ Prediction failed: {error}")
            st.stop()
        
        # Display results
        prediction = result['prediction']
        enrichment = result['enrichment']
        explanation = result.get('explanation')
        metadata = result['metadata']
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delay_status = "🔴 YES" if prediction['will_delay'] else "🟢 NO"
            st.metric("Will Delay?", delay_status)
        
        with col2:
            st.metric(
                "Delay Probability",
                f"{prediction['delay_probability']:.1%}",
                delta=f"{enrichment['vendor_adjustment']:+.1%} (enrichment)"
            )
        
        with col3:
            st.metric("Estimated Delay", f"{prediction['estimated_delay_days']:.1f} days")
        
        with col4:
            st.metric("Delay Reason", prediction['delay_reason'])
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_probability_gauge(prediction['delay_probability']),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_enrichment_waterfall(
                    enrichment['raw_model_probability'],
                    enrichment['vendor_adjustment'],
                    prediction['delay_probability']
                ),
                use_container_width=True
            )
        
        st.divider()
        
        # Enrichment Details
        st.subheader("🔧 Enrichment Layer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Vendor Intelligence")
            vendor_tier_color = {
                "EXCELLENT": "🟢",
                "GOOD": "🟡",
                "AVERAGE": "🟠",
                "POOR": "🔴"
            }
            tier_emoji = vendor_tier_color.get(enrichment['vendor_tier'], "⚪")
            
            st.markdown(f"""
            - **Vendor**: {metadata['vendor']}
            - **Reliability Tier**: {tier_emoji} {enrichment['vendor_tier']}
            - **On-Time Rate**: {enrichment['vendor_on_time_rate']:.1%}
            - **Impact**: {enrichment['vendor_adjustment']:+.1%}
            """)
        
        with col2:
            st.markdown("### Route Intelligence")
            st.markdown(f"""
            - **Route**: {metadata['route']}
            - **Historical Delay Rate**: {enrichment['route_historical_delay']:.1%}
            - **Confidence**: {enrichment['route_confidence']:.1%}
            - **Base Model**: {enrichment['raw_model_probability']:.1%}
            """)
        
        st.divider()
        
        # LLM Explanation
        if explanation:
            st.subheader("💡 AI-Generated Explanation")
            
            if prediction['will_delay']:
                st.markdown(f'<div class="warning-card">{explanation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card">{explanation}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Technical Details (Expandable)
        with st.expander("🔍 Technical Details"):
            st.json(result)
    
    else:
        # Welcome screen
        st.info("👈 Fill in the shipment details in the sidebar and click **Predict Delay** to get started!")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Ensemble ML Models
            - CatBoost
            - XGBoost  
            - LightGBM
            - 87%+ Accuracy
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Smart Enrichment
            - Vendor reliability tracking
            - Route pattern analysis
            - 50 vendors, 2,449 routes
            - Real-time adjustments
            """)
        
        with col3:
            st.markdown("""
            ### 💡 LLM Explanations
            - Natural language insights
            - Powered by Groq
            - Actionable recommendations
            - Context-aware analysis
            """)


if __name__ == "__main__":
    main()
