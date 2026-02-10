"""
HuggingFace API-based LLM explainer for generating clinical explanations.
"""

from huggingface_hub import InferenceClient
from typing import Dict, Any, List, Optional
import time


class HuggingFaceExplainer:
    """Generate clinical explanations using HuggingFace Inference API."""
    
    def __init__(
        self,
        api_token: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens: int = 300,
        temperature: float = 0.7,
        timeout: int = 60
    ):
        """
        Initialize HuggingFace explainer using the official client library.
        
        Args:
            api_token: HuggingFace API token
            model_name: Name of the LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: API request timeout in seconds
        
        Recommended models (serverless inference):
            - meta-llama/Meta-Llama-3-8B-Instruct (default, chat completion)
            - microsoft/Phi-3-mini-4k-instruct (fast, reliable)
            - google/gemma-7b-it (alternative option)
        """
        self.api_token = api_token
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize the official HuggingFace Inference Client
        self.client = InferenceClient(token=api_token)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    def _create_prompt(
        self,
        risk_level: str,
        anomaly_score: float,
        feature_importances: Dict[str, float],
        feature_values: Dict[str, float],
        model_name: str
    ) -> str:
        """
        Create structured prompt for the LLM.
        
        Args:
            risk_level: Risk level (Low/Medium/High)
            anomaly_score: Normalized anomaly score [0, 1]
            feature_importances: Dict of feature names to importance scores
            feature_values: Dict of feature names to actual values
            model_name: Name of the anomaly detection model
            
        Returns:
            Formatted prompt string
        """
        # Format features with both actual values and importance scores
        feature_lines = []
        for feature, importance in feature_importances.items():
            actual_value = feature_values.get(feature, "N/A")
            if isinstance(actual_value, (int, float)):
                feature_lines.append(f"- {feature}: {actual_value:.2f} (importance: {importance:.3f})")
            else:
                feature_lines.append(f"- {feature}: {actual_value} (importance: {importance:.3f})")
        feature_text = "\n".join(feature_lines)
        
        # Create prompt for instruction-tuned models
        prompt = f"""<s>[INST] You are a medical AI assistant.

Analyze this heart attack risk assessment:
- Risk Level: {risk_level}
- Anomaly Score: {anomaly_score:.3f} (0=normal, 1=high risk)
- Detection Model: {model_name}

Patient Key Features (with importance scores):
{feature_text}

Write a brief 2-3 sentence clinical explanation for a cardiologist. Focus on the abnormal features and their actual values. Do not diagnose or provide medical advice. [/INST]</s>"""
        
        return prompt
    
    def _call_api(self, prompt: str, retry_count: int = 3) -> str:
        """
        Call HuggingFace Inference API using the official client.
        
        Args:
            prompt: Input prompt
            retry_count: Number of retries on failure
            
        Returns:
            Generated text (or error message if failed)
        """
        for attempt in range(retry_count):
            try:
                # Try using chat completion interface first (newer API)
                try:
                    response = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        if content:
                            return content.strip()
                    
                except AttributeError:
                    # If chat_completion doesn't work, try text_generation
                    response = self.client.text_generation(
                        prompt,
                        model=self.model_name,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        return_full_text=False
                    )
                    
                    if response and isinstance(response, str):
                        return response.strip()
                
                return "No explanation generated"
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Model is loading
                if "503" in error_msg or "loading" in error_msg:
                    wait_time = 20 + (10 * attempt)
                    print(f"Model loading... waiting {wait_time}s (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                    continue
                
                # Rate limiting
                elif "429" in error_msg or "rate" in error_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Authorization error
                elif "401" in error_msg or "403" in error_msg or "authorization" in error_msg:
                    return "Error: Invalid HuggingFace API token. Get a new token at https://huggingface.co/settings/tokens"
                
                # Model not found or not supported
                elif "404" in error_msg or "not found" in error_msg or "not supported" in error_msg:
                    return f"Error: Model '{self.model_name}' not available. Try: 'meta-llama/Meta-Llama-3-8B-Instruct' or 'microsoft/Phi-3-mini-4k-instruct'"
                
                # Model requires agreement
                elif "gated" in error_msg or "agreement" in error_msg:
                    return f"Error: Model '{self.model_name}' requires accepting terms at https://huggingface.co/{self.model_name}"
                
                # Generic error
                else:
                    print(f"Attempt {attempt + 1}/{retry_count}: {type(e).__name__}: {str(e)[:100]}")
                    
                    if attempt == retry_count - 1:
                        return f"Error: {type(e).__name__}: {str(e)[:200]}"
                    
                    time.sleep(3 * (attempt + 1))
        
        return "Error: Failed to generate explanation after all retries"
    
    def explain(
        self,
        risk_level: str,
        anomaly_score: float,
        feature_importances: Dict[str, float],
        feature_values: Dict[str, float],
        model_name: str = "Anomaly Detection Model"
    ) -> Dict[str, Any]:
        """
        Generate clinical explanation for anomaly detection result.
        
        Args:
            risk_level: Risk level (Low/Medium/High)
            anomaly_score: Normalized anomaly score [0, 1]
            feature_importances: Dict of feature names to importance scores
            feature_values: Dict of feature names to actual values
            model_name: Name of the anomaly detection model
            
        Returns:
            Dict containing explanation and metadata
        """
        # Create prompt
        prompt = self._create_prompt(
            risk_level=risk_level,
            anomaly_score=anomaly_score,
            feature_importances=feature_importances,
            feature_values=feature_values,
            model_name=model_name
        )
        
        # Call API
        explanation_text = self._call_api(prompt)
        
        return {
            "explanation": explanation_text,
            "risk_level": risk_level,
            "anomaly_score": anomaly_score,
            "model_name": model_name,
            "top_features": list(feature_importances.keys())
        }
    
    def batch_explain(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple samples.
        
        Args:
            results: List of dicts with risk_level, anomaly_score, feature_importances, feature_values
            
        Returns:
            List of explanation dicts
        """
        explanations = []
        
        for i, result in enumerate(results):
            print(f"Generating explanation {i+1}/{len(results)}...")
            explanation = self.explain(
                risk_level=result["risk_level"],
                anomaly_score=result["anomaly_score"],
                feature_importances=result["feature_importances"],
                feature_values=result.get("feature_values", {}),
                model_name=result.get("model_name", "Anomaly Detection Model")
            )
            explanations.append(explanation)
            
            # Small delay to avoid rate limiting
            if i < len(results) - 1:
                time.sleep(1)
        
        return explanations
    
    def generate_comparative_summary(
        self,
        model_results: Dict[str, Dict[str, Any]],
        feature_values: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate a comparative summary across multiple models.
        
        Args:
            model_results: Dict mapping model names to their results
            feature_values: Dict of actual feature values for the patient (optional)
            
        Returns:
            Comparative summary text
        """
        # Create summary prompt
        model_summaries = []
        for model_name, result in model_results.items():
            model_summaries.append(
                f"- {model_name}: Risk={result['risk_level']}, Score={result['anomaly_score']:.3f}"
            )
        
        summary_text = "\n".join(model_summaries)
        
        # Add patient feature values if available
        patient_context = ""
        if feature_values:
            feature_lines = []
            # Show top 5 most relevant features
            for i, (feature, value) in enumerate(list(feature_values.items())[:5]):
                if isinstance(value, (int, float)):
                    feature_lines.append(f"  - {feature}: {value:.2f}")
                else:
                    feature_lines.append(f"  - {feature}: {value}")
            
            if feature_lines:
                patient_context = f"\n\nPatient Data (sample features):\n" + "\n".join(feature_lines)
        
        prompt = f"""<s>[INST] Multiple anomaly detection models analyzed a patient for heart attack risk:

{summary_text}{patient_context}

Write 2-3 sentences synthesizing these results. Note if models agree or disagree. Use clinical language. Do not diagnose or provide medical advice. [/INST]</s>"""
        
        result = self._call_api(prompt)
        
        # Return the result or a fallback message
        if result and not result.startswith("Error"):
            return result
        else:
            return "Unable to generate comparative summary. " + (result if result else "")
    
    def validate_api_connection(self) -> Dict[str, Any]:
        """
        Validate API connection and token.
        
        Returns:
            Dict with validation status and details
        """
        try:
            # Try chat completion interface first (newer API)
            try:
                test_response = self.client.chat_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    model=self.model_name,
                    max_tokens=10
                )
                
                return {
                    "connected": True,
                    "status_code": 200,
                    "model": self.model_name,
                    "message": "✓ Connected successfully (chat API)"
                }
            except:
                # Fallback to text generation
                test_response = self.client.text_generation(
                    "Test",
                    model=self.model_name,
                    max_new_tokens=10
                )
                
                return {
                    "connected": True,
                    "status_code": 200,
                    "model": self.model_name,
                    "message": "✓ Connected successfully (text gen API)"
                }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "401" in error_msg or "403" in error_msg or "authorization" in error_msg:
                message = "✗ Invalid API token"
            elif "404" in error_msg or "not found" in error_msg:
                message = f"✗ Model '{self.model_name}' not found"
            elif "503" in error_msg or "loading" in error_msg:
                message = "✓ Connected (model loading)"
                return {
                    "connected": True,
                    "status_code": 503,
                    "model": self.model_name,
                    "message": message
                }
            elif "gated" in error_msg:
                message = f"✗ Model requires accepting terms at https://huggingface.co/{self.model_name}"
            else:
                message = f"✗ Error: {str(e)[:100]}"
            
            return {
                "connected": False,
                "status_code": 0,
                "model": self.model_name,
                "message": message
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model being used.
        
        Returns:
            Dict containing model metadata
        """
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
