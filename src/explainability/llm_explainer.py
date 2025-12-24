"""
LLM-based natural language explanation generator using LangChain + Groq.

Uses Groq's LLaMA models via LangChain to generate human-readable explanations
of delay predictions based on model outputs and enrichment context.
"""

from typing import Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


class LLMExplainer:
    """Generate natural language explanations using LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM explainer with LangChain + Groq.
        
        Args:
            api_key: Groq API key. Uses settings if None.
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("No Groq API key found. Explanations will be unavailable.")
            self.llm = None
        else:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,  # Low for factual consistency
                max_tokens=300,
                groq_api_key=self.api_key
            )
            logger.info("LLM explainer initialized with LangChain + Groq")
    
    def generate_explanation(
        self,
        prediction: Dict,
        enrichment: Dict,
        shipment_data: Dict
    ) -> str:
        """
        Generate natural language explanation for prediction.
        
        Args:
            prediction: Prediction results
            enrichment: Enrichment layer details
            shipment_data: Original shipment data
        
        Returns:
            Human-readable explanation string
        """
        if not self.llm:
            return "LLM explanations unavailable - API key not configured"
        
        try:
            # Build structured prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(prediction, enrichment, shipment_data)
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Invoke LLM
            response = self.llm.invoke(messages)
            explanation = response.content.strip()
            
            logger.info("Generated LLM explanation (%d chars)", len(explanation))
            return explanation
            
        except Exception as exc:
            logger.exception("Failed to generate LLM explanation")
            return f"Explanation generation failed: {str(exc)}"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for consistent explanation style."""
        return """You are FlowSight AI, an expert supply chain analyst explaining shipment delay predictions to logistics managers.

Your explanations should:
1. Be clear, concise, and actionable (3-4 sentences max)
2. Start with the bottom line (will it delay or not?)
3. Explain the key factors driving the prediction
4. Provide context using specific numbers and percentages
5. End with a practical insight or recommendation

Tone: Professional but conversational. Avoid jargon. Use "this shipment" not "the shipment".

Example style:
"This shipment has a 72% chance of delay, primarily due to the vendor's poor reliability track record (only 45% on-time rate). The Mumbai→Delhi route also shows a 64% historical delay rate, reinforcing the risk. The model initially predicted 65% delay probability, which increased to 72% after factoring in vendor performance. Recommendation: Consider expedited shipping or alternative vendor for time-sensitive orders."
"""
    
    def _build_user_prompt(
        self,
        prediction: Dict,
        enrichment: Dict,
        shipment_data: Dict
    ) -> str:
        """Build structured prompt with prediction data."""
        
        # Extract key information
        will_delay = prediction['will_delay']
        delay_prob = prediction['delay_probability']
        delay_days = prediction['estimated_delay_days']
        delay_reason = prediction['delay_reason']
        
        vendor_tier = enrichment['vendor_tier']
        vendor_otr = enrichment['vendor_on_time_rate']
        route_hist = enrichment['route_historical_delay']
        raw_prob = enrichment['raw_model_probability']
        vendor_adj = enrichment['vendor_adjustment']
        
        route = f"{shipment_data.get('origin_city', '')}→{shipment_data.get('destination_city', '')}"
        vendor = shipment_data.get('supplier_name', 'Unknown')
        product = shipment_data.get('product_category', 'Unknown')
        weight = shipment_data.get('weight_kg', 0)
        value = shipment_data.get('value_inr', 0)
        
        prompt = f"""Explain this shipment delay prediction:

**SHIPMENT:**
- Route: {route}
- Vendor: {vendor} (Tier: {vendor_tier}, On-time Rate: {vendor_otr:.1%})
- Product: {product}, Weight: {weight:.0f}kg, Value: ₹{value:,.0f}

**PREDICTION:**
- Delay Expected: {'YES' if will_delay else 'NO'}
- Delay Probability: {delay_prob:.1%}
- Estimated Delay: {delay_days:.1f} days
- Primary Reason: {delay_reason}

**ANALYSIS:**
- Base Model: {raw_prob:.1%} delay probability
- Vendor Impact: {vendor_adj:+.1%} adjustment
- Route History: {route_hist:.1%} past delay rate

Generate a professional 3-4 sentence explanation for the logistics manager."""

        return prompt
    
    def generate_batch_explanations(
        self,
        predictions_batch: list[Dict]
    ) -> list[str]:
        """
        Generate explanations for multiple predictions in batch.
        
        Args:
            predictions_batch: List of prediction dictionaries
        
        Returns:
            List of explanation strings
        """
        if not self.llm:
            return ["LLM unavailable"] * len(predictions_batch)
        
        explanations = []
        for pred_data in predictions_batch:
            explanation = self.generate_explanation(
                pred_data['prediction'],
                pred_data['enrichment'],
                pred_data['shipment_data']
            )
            explanations.append(explanation)
        
        return explanations
