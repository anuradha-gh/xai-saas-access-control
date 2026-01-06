"""
Enhanced Explanation Templates for XAI
Provides context-aware, risk-level specific, and actionable explanations
"""

from typing import Dict, Any, List


class EnhancedTemplateGenerator:
    """Generate better explanation templates based on context"""
    
    def __init__(self):
        pass
    
    def generate_c1_explanation(self, risk_score: int, category: str, 
                                shap_values: List[Dict], log_data: Dict) -> str:
        """
        Generate context-aware C1 (Anomaly Detection) explanation
        
        Args:
            risk_score: Risk score (0-100)
            category: Risk category (LOW/MEDIUM/HIGH)
            shap_values: SHAP feature importance list
            log_data: Original log context
        
        Returns:
            Professional explanation with actionable recommendations
        """
        event_name = log_data.get('eventName', 'Unknown action')
        user_type = log_data.get('userIdentityType', 'Unknown user')
        service = log_data.get('eventSource', 'Unknown service')
        
        # Get top contributing feature
        if shap_values and len(shap_values) > 0:
            top_feature = shap_values[0]
            feature_name = top_feature['feature']
            importance = abs(top_feature.get('importance', 0))
            
            # Interpret feature
            feature_explanation = self._interpret_c1_feature(feature_name, importance)
        else:
            feature_explanation = "unusual activity pattern detected"
        
        # Risk-level specific messaging
        if category == "HIGH":
            severity = "🔴 **CRITICAL ALERT**"
            action = "**Immediate investigation required**"
            explanation = f"""
{severity}
Risk Score: {risk_score}/100 (High Risk)

Activity: {user_type} performed '{event_name}' on {service}

Primary Concern: {feature_explanation.capitalize()}.

Why This Matters:
- This activity pattern is {importance*100:.0f}% different from established normal behavior
- High-risk actions by {user_type} accounts require immediate verification
- Potential indicators of compromised credentials or unauthorized access

{action}:
1. Verify if this action was intentional and authorized
2. Review recent access patterns for this account
3. Check for other unusual activities in the same timeframe
4. Consider requiring MFA reauthentication
                """.strip()
        
        elif category == "MEDIUM":
            severity = "⚠️ **MODERATE RISK**"
            action = "Review recommended"
            explanation = f"""
{severity}
Risk Score: {risk_score}/100 (Medium Risk)

Activity: {user_type} performed '{event_name}' on {service}

Detected Anomaly: {feature_explanation.capitalize()}.

Assessment:
- Activity shows {importance*100:.0f}% deviation from typical patterns
- Not immediately critical but warrants investigation
- May indicate configuration changes or policy updates

Recommended Actions:
1. Review if this aligns with recent authorized changes
2. Document the activity for audit purposes
3. Monitor for similar patterns in the next 24-48 hours
                """.strip()
        
        else:  # LOW
            severity = "ℹ️ **INFORMATIONAL**"
            action = "Monitoring only"
            explanation = f"""
{severity}
Risk Score: {risk_score}/100 (Low Risk)

Activity: {user_type} performed '{event_name}' on {service}

Observation: {feature_explanation.capitalize()}.

Context:
- Minor deviation ({importance*100:.0f}%) from normal patterns detected
- Likely routine activity with slight variation
- No immediate concern

Action: {action} - logged for trend analysis
                """.strip()
        
        return explanation
    
    def _interpret_c1_feature(self, feature_name: str, importance: float) -> str:
        """Interpret C1 SHAP feature for users"""
        
        interpretations = {
            'mse': f"reconstruction error indicates {importance*100:.0f}% abnormal activity pattern",
            'latent_0': f"user behavior pattern shows {importance*100:.0f}% anomaly",
            'latent_1': f"temporal access pattern deviates {importance*100:.0f}%",
            'latent_2': f"resource access sequence is {importance*100:.0f}% unusual",
            'latent_3': f"API call frequency differs {importance*100:.0f}%",
            'latent_4': f"service interaction pattern is {importance*100:.0f}% atypical",
            'latent_5': f"API call sequence shows {importance*100:.0f}% deviation",
            'latent_6': f"geographic access pattern is {importance*100:.0f}% irregular",
            'latent_7': f"time-of-day activity differs {importance*100:.0f}%",
        }
        
        return interpretations.get(feature_name, f"activity pattern shows {importance*100:.0f}% anomaly")
    
    def generate_c2_explanation(self, predicted_role: str, confidence: float,
                                lime_words: List[Dict], log_text: str) -> str:
        """
        Generate C2 (Role Classification) explanation
        
        Args:
            predicted_role: Predicted role label
            confidence: Confidence score (0-1)
            lime_words: LIME word importance
            log_text: Original log text
        
        Returns:
            Role recommendation with justification
        """
        conf_pct = confidence * 100
        
        # Extract key words
        if lime_words and len(lime_words) > 0:
            positive_words = [w['word'] for w in lime_words[:3] if w.get('importance', 0) > 0]
            key_indicators = ", ".join(f"'{w}'" for w in positive_words) if positive_words else "activity pattern"
        else:
            key_indicators = "activity pattern"
        
        # Confidence-based messaging
        if conf_pct >= 80:
            confidence_level = "High"
            recommendation = "recommended"
        elif conf_pct >= 60:
            confidence_level = "Moderate"
            recommendation = "suggested"
        else:
            confidence_level = "Low"
            recommendation = "tentative"
        
        explanation = f"""
👤 **ROLE CLASSIFICATION**

Recommended Role: {predicted_role}
Confidence Level: {confidence_level} ({conf_pct:.1f}%)

Analysis:
The classification is based on keywords {key_indicators} which are strong indicators 
of {predicted_role} activities.

Recommendation: This account is {recommendation} for '{predicted_role}' role assignment.

Implications:
- Role determines access permissions and resource limits
- Ensure assigned role aligns with user's actual responsibilities
- Review periodically to maintain principle of least privilege

Next Steps:
{'1. Approve role assignment' if conf_pct >= 80 else '1. Review role assignment manually'}
2. Document role justification for audit trail
            """.strip()
        
        return explanation
    
    def generate_c3_explanation(self, decision: str, confidence: int,
                                similar_activities: List[Dict]) -> str:
        """
        Generate C3 (Access Decision) explanation
        
        Args:
            decision: Access decision (APPROVE/DENY/MFA)
            confidence: Confidence score (0-10)
            similar_activities: List of similar past activities
        
        Returns:
            Access decision explanation
        """
        # Decision-specific messaging
        if decision == "APPROVE":
            icon = "✅"
            action = "GRANTED"
            color = "green"
        elif decision == "DENY":
            icon = "❌"
            action = "DENIED"
            color = "red"
        else:  # MFA/OTHER
            icon = "🔐"
            action = "REQUIRES MFA"
            color = "orange"
        
        # Similar activities context
        if similar_activities and len(similar_activities) > 0:
            top_similar = similar_activities[0]
            similarity = top_similar.get('similarity', 0) * 100
            context = f"This action is {similarity:.0f}% similar to: '{top_similar['text'][:80]}...'"
        else:
            context = "No similar past activities found"
        
        explanation = f"""
🔐 **ACCESS DECISION**

Decision: {icon} {action}
Confidence: {confidence}/10

Semantic Analysis:
{context}

Rationale:
- Decision based on similarity to known patterns and security policies
- Confidence score indicates system's certainty in this decision
- Higher similarity to normal patterns → more likely to approve

Implementation:
{'- Access granted automatically' if decision == 'APPROVE' else '- User prompted for additional verification' if 'MFA' in decision else '- Access blocked pending manual review'}

Security Note:
All access decisions are logged and auditable for compliance purposes.
            """.strip()
        
        return explanation


# Example usage:
"""
from xai_templates import EnhancedTemplateGenerator

template_gen = EnhancedTemplateGenerator()

# In llm_translator.py fallback:
if not self.ollama_available:
    return template_gen.generate_c1_explanation(
        risk_score=85,
        category='HIGH',
        shap_values=shap_data['feature_importance'],
        log_data={'eventName': 'DeleteBucket', 'userIdentityType': 'Root'}
    )
"""
