"""
LLM Translation Layer
Converts numerical XAI outputs into natural language explanations
tailored for different stakeholder types (technical, executive, compliance)
"""

import json
from typing import Dict, Any, List, Optional
from enum import Enum
from xai_templates import EnhancedTemplateGenerator  # IMPROVEMENT #4


class StakeholderType(Enum):
    """Types of stakeholders with different explanation needs"""
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    COMPLIANCE = "compliance"
    GENERAL = "general"


class LLMTranslator:
    """Translates XAI numerical outputs to natural language"""
    
    def __init__(self, use_ollama: bool = True, model_name: str = "gemma3:1b"):
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.ollama_available = False
        
        # IMPROVEMENT #4: Initialize enhanced template generator
        self.template_gen = EnhancedTemplateGenerator()
        
        if self.use_ollama:
            try:
                import ollama
                self.ollama = ollama
                self.ollama_available = True
                print("✓ LLM Translator initialized with Ollama")
            except ImportError:
                print("⚠️ Ollama not available - using enhanced template explanations")
                self.ollama_available = False
    
    def _build_system_prompt(self, stakeholder: StakeholderType) -> str:
        """Build system prompt based on stakeholder type"""
        
        prompts = {
            StakeholderType.TECHNICAL: """You are a Senior Machine Learning Engineer specializing in Explainable AI. 
Provide detailed technical explanations including model architectures, feature importance metrics, and statistical measures.
Use precise technical terminology. Target audience: ML engineers and data scientists.""",
            
            StakeholderType.EXECUTIVE: """You are a Chief Information Security Officer (CISO) communicating with executive leadership.
Provide high-level summaries focused on business impact, risk levels, and actionable decisions.
Avoid technical jargon. Use clear business language. Target audience: C-suite executives.""",
            
            StakeholderType.COMPLIANCE: """You are a Compliance and Audit Specialist.
Provide explanations focused on regulatory requirements, audit trails, and policy adherence.
Highlight potential compliance violations and remediation steps.
Target audience: Compliance officers and auditors.""",
            
            StakeholderType.GENERAL: """You are a Cybersecurity Analyst.
Provide balanced explanations that are accurate but accessible to non-experts.
Use analogies when helpful. Target audience: General security team members."""
        }
        
        return prompts.get(stakeholder, prompts[StakeholderType.GENERAL])
    
    def _build_c1_prompt(self, xai_data: Dict[str, Any], stakeholder: StakeholderType) -> str:
        """Build prompt for C1 (Anomaly Detection) explanation"""
        
        risk_score = xai_data.get('risk_score', 0)
        category = xai_data.get('category', 'UNKNOWN')
        shap_data = xai_data.get('shap', {})
        recon_data = xai_data.get('reconstruction', {})
        log_data = xai_data.get('log_context', {})
        
        context = f"""
Component: Anomaly Detection (C1)
Risk Score: {risk_score}/100 ({category})
Log Details:
  - Event: {log_data.get('eventName', 'Unknown')}
  - Service: {log_data.get('eventSource', 'Unknown')}
  - User: {log_data.get('userName', 'Unknown')}
  - Region: {log_data.get('awsRegion', 'Unknown')}

XAI Analysis:
"""
        
        # Add SHAP information
        if 'feature_importance' in shap_data:
            context += "\nFeature Importance (SHAP):\n"
            for feat in shap_data['feature_importance'][:5]:
                context += f"  - {feat['feature']}: {feat['shap_value']:.3f} (importance: {feat['importance']:.3f})\n"
        
        # Add reconstruction errors
        if 'feature_errors' in recon_data:
            context += "\nReconstruction Errors:\n"
            for feat in recon_data['feature_errors'][:3]:
                context += f"  - {feat['feature']}: {feat['error']:.3f} (value: {feat['original_value']})\n"
        
        if stakeholder == StakeholderType.TECHNICAL:
            context += "\nProvide a technical explanation of why this log has this risk score. Include:\n"
            context += "1. Which features contribute most to the anomaly\n"
            context += "2. How reconstruction error indicates abnormality\n"
            context += "3. Statistical significance of the anomaly score\n"
        elif stakeholder == StakeholderType.EXECUTIVE:
            context += "\nProvide an executive summary in 2-3 sentences:\n"
            context += "1. What action was flagged and why\n"
            context += "2. Business impact if this is malicious\n"
            context += "3. Recommended next steps\n"
        elif stakeholder == StakeholderType.COMPLIANCE:
            context += "\nProvide a compliance-focused explanation:\n"
            context += "1. Which policy rules are triggered\n"
            context += "2. Potential regulatory violations\n"
            context += "3. Required audit actions\n"
        else:
            context += "\nExplain this anomaly in simple terms (2-3 sentences).\n"
        
        context += "\nKeep the response under 100 words. Do not use markdown formatting."
        
        return context
    
    def _build_c2_prompt(self, xai_data: Dict[str, Any], stakeholder: StakeholderType) -> str:
        """Build prompt for C2 (Role Classification) explanation"""
        
        predicted_role = xai_data.get('predicted_role', 'Unknown')
        confidence = xai_data.get('confidence', 0)
        lime_data = xai_data.get('lime', {})
        attention_data = xai_data.get('attention', {})
        log_data = xai_data.get('log_context', {})
        
        context = f"""
Component: Role Classification (C2)
Predicted Role: {predicted_role} (confidence: {confidence:.2%})
Log Text: {log_data.get('text', 'N/A')}

XAI Analysis:
"""
        
        # Add LIME token importance
        if 'word_importance' in lime_data:
            context += "\nImportant Words (LIME):\n"
            for word in lime_data['word_importance'][:5]:
                context += f"  - '{word['word']}': {word['importance']:.3f} ({word['direction']})\n"
        
        # Add attention weights
        if 'token_attention' in attention_data:
            context += "\nHighest Attention Tokens:\n"
            for token in attention_data['token_attention'][:5]:
                context += f"  - '{token['token']}': {token['attention']:.3f}\n"
        
        if stakeholder == StakeholderType.TECHNICAL:
            context += "\nProvide technical explanation:\n"
            context += "1. Why BERT classified this as the predicted role\n"
            context += "2. Key tokens driving the decision\n"
            context += "3. Model confidence interpretation\n"
        elif stakeholder == StakeholderType.EXECUTIVE:
            context += "\nProvide executive summary:\n"
            context += "1. What role is recommended\n"
            context += "2. Why this matters for access control\n"
            context += "3. Risk if role is incorrect\n"
        elif stakeholder == StakeholderType.COMPLIANCE:
            context += "\nProvide compliance explanation:\n"
            context += "1. Role assignment justification\n"
            context += "2. Principle of least privilege compliance\n"
            context += "3. Audit trail requirements\n"
        else:
            context += "\nExplain the role prediction in simple terms (2-3 sentences).\n"
        
        context += "\nKeep under 100 words. No markdown."
        
        return context
    
    def _build_c3_prompt(self, xai_data: Dict[str, Any], stakeholder: StakeholderType) -> str:
        """Build prompt for C3 (Access Decision) explanation"""
        
        decision = xai_data.get('decision', 'UNKNOWN')
        confidence = xai_data.get('confidence', 0)
        embedding_data = xai_data.get('embedding', {})
        log_data = xai_data.get('log_context', {})
        
        context = f"""
Component: Access Decision (C3)
Decision: {decision} (confidence: {confidence}/10)
Log: {log_data.get('text', 'N/A')}

XAI Analysis:
"""
        
        # Add semantic neighbors
        if 'nearest_neighbors' in embedding_data:
            context += "\nSimilar Past Activities:\n"
            for neighbor in embedding_data['nearest_neighbors'][:3]:
                context += f"  - '{neighbor['text'][:60]}...' (similarity: {neighbor['similarity']:.2%})\n"
        
        # Add isolation score
        if 'isolation_score' in embedding_data:
            context += f"\nSemantic Anomaly Score: {embedding_data['isolation_score']:.3f}\n"
        
        # Add important dimensions
        if 'important_dimensions' in embedding_data:
            context += "\nKey Embedding Dimensions:\n"
            for dim in embedding_data['important_dimensions'][:3]:
                context += f"  - Dimension {dim['dimension']}: {dim['value']:.3f} (importance: {dim['importance']:.3f})\n"
        
        if stakeholder == StakeholderType.TECHNICAL:
            context += "\nProvide technical explanation:\n"
            context += "1. How semantic similarity influences decision\n"
            context += "2. Why specific decision (approve/deny/MFA) was chosen\n"
            context += "3. Confidence score interpretation\n"
        elif stakeholder == StakeholderType.EXECUTIVE:
            context += "\nProvide executive summary:\n"
            context += "1. What decision was made and why\n"
            context += "2. Business continuity vs security tradeoff\n"
            context += "3. Expected outcome\n"
        elif stakeholder == StakeholderType.COMPLIANCE:
            context += "\nProvide compliance explanation:\n"
            context += "1. Policy-based decision rationale\n"
            context += "2. Audit logging sufficiency\n"
            context += "3. Override conditions\n"
        else:
            context += "\nExplain the access decision in simple terms (2-3 sentences).\n"
        
        context += "\nKeep under 100 words. No markdown."
        
        return context
    
    def _fallback_explanation(self, component: str, xai_data: Dict[str, Any]) -> str:
        """IMPROVEMENT #4: Enhanced template-based fallback"""
        
        if component == "C1":
            risk_score = xai_data.get('risk_score', 0)
            category = xai_data.get('category', 'UNKNOWN')
            shap_data = xai_data.get('shap', {})
            shap_values = shap_data.get('feature_importance', [])
            log_context = xai_data.get('log_context', {})
            
            return self.template_gen.generate_c1_explanation(
                risk_score, category, shap_values, log_context
            )
        
        elif component == "C2":
            predicted_role = xai_data.get('predicted_role', 'Unknown')
            confidence = xai_data.get('confidence', 0)
            lime_data = xai_data.get('lime', {})
            lime_words = lime_data.get('word_importance', [])
            log_context = xai_data.get('log_context', {})
            log_text = log_context.get('text', '')
            
            return self.template_gen.generate_c2_explanation(
                predicted_role, confidence, lime_words, log_text
            )
        
        elif component == "C3":
            decision = xai_data.get('decision', 'UNKNOWN')
            confidence = xai_data.get('confidence', 0)
            embedding_data = xai_data.get('embedding', {})
            similar_activities = embedding_data.get('nearest_neighbors', [])
            
            return self.template_gen.generate_c3_explanation(
                decision, confidence, similar_activities
            )
        
        return f"[Component {component}] XAI data: {xai_data}"
    
    def translate(self, 
                  component: str, 
                  xai_data: Dict[str, Any], 
                  stakeholder: StakeholderType = StakeholderType.GENERAL) -> str:
        """
        Translate XAI numerical data to natural language
        
        Args:
            component: "C1", "C2", or "C3"
            xai_data: Dictionary containing XAI analysis results
            stakeholder: Target audience type
        
        Returns:
            Natural language explanation string
        """
        
        # Fallback if LLM not available
        if not self.ollama_available:
            return self._fallback_explanation(component, xai_data)
        
        # Build prompt based on component
        if component == "C1":
            user_prompt = self._build_c1_prompt(xai_data, stakeholder)
        elif component == "C2":
            user_prompt = self._build_c2_prompt(xai_data, stakeholder)
        elif component == "C3":
            user_prompt = self._build_c3_prompt(xai_data, stakeholder)
        else:
            return f"Unknown component: {component}"
        
        # Get system prompt
        system_prompt = self._build_system_prompt(stakeholder)
        
        # Call LLM
        try:
            response = self.ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            explanation = response['message']['content'].strip()
            
            # Remove markdown formatting if present
            explanation = explanation.replace('**', '').replace('*', '')
            
            return explanation
            
        except Exception as e:
            print(f"⚠️ LLM translation error: {e}")
            return self._fallback_explanation(component, xai_data)
    
    def translate_comparative(self, 
                             component: str,
                             option_a: Dict[str, Any],
                             option_b: Dict[str, Any],
                             stakeholder: StakeholderType = StakeholderType.GENERAL) -> str:
        """
        Generate comparative explanation: "Why A instead of B?"
        
        Args:
            component: Component name
            option_a: The chosen option with XAI data
            option_b: The alternative option with XAI data
            stakeholder: Target audience
        
        Returns:
            Comparative explanation
        """
        
        if not self.ollama_available:
            return "Comparative explanations require LLM support."
        
        system_prompt = self._build_system_prompt(stakeholder)
        
        user_prompt = f"""
Compare two options and explain why Option A was chosen over Option B.

Option A (Chosen):
{json.dumps(option_a, indent=2)}

Option B (Alternative):
{json.dumps(option_b, indent=2)}

Explain in 2-3 sentences:
1. Key differences between options
2. Why A is preferred over B
3. What would need to change for B to be preferred

Keep under 100 words. No markdown.
"""
        
        try:
            response = self.ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            return response['message']['content'].strip().replace('**', '').replace('*', '')
            
        except Exception as e:
            return f"Comparison failed: {e}"
    
    def generate_report(self, 
                       full_analysis: Dict[str, Any],
                       stakeholder: StakeholderType = StakeholderType.GENERAL) -> str:
        """
        Generate comprehensive report for all three components
        
        Args:
            full_analysis: Dict with 'c1', 'c2', 'c3' XAI results
            stakeholder: Target audience
        
        Returns:
            Full formatted report
        """
        
        report = f"\n{'='*70}\n"
        report += f"XAI ANALYSIS REPORT - {stakeholder.value.upper()} VIEW\n"
        report += f"{'='*70}\n\n"
        
        # Component 1
        if 'c1' in full_analysis:
            report += "🔍 ANOMALY DETECTION\n"
            report += "-" * 70 + "\n"
            report += self.translate("C1", full_analysis['c1'], stakeholder) + "\n\n"
        
        # Component 2
        if 'c2' in full_analysis:
            report += "👤 ROLE CLASSIFICATION\n"
            report += "-" * 70 + "\n"
            report += self.translate("C2", full_analysis['c2'], stakeholder) + "\n\n"
        
        # Component 3
        if 'c3' in full_analysis:
            report += "🔐 ACCESS DECISION\n"
            report += "-" * 70 + "\n"
            report += self.translate("C3", full_analysis['c3'], stakeholder) + "\n\n"
        
        report += "=" * 70 + "\n"
        
        return report
