"""
LLM interface module for the RAG Chatbot system.
Provides a unified interface for different language models.
"""

import os
import torch
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        AutoModelForSeq2SeqLM, pipeline
    )
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")

class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass

class HuggingFaceLLM(BaseLLM):
    """Interface for Hugging Face language models."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 model_type: str = "auto",
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name: Name of the model on Hugging Face Hub
            model_type: Type of model ("auto", "causal", "seq2seq")
            device: Device to run the model on ("cpu", "cuda", "auto")
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Model parameters
        self.max_length = kwargs.get("max_length", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        self.do_sample = kwargs.get("do_sample", True)
        self.pad_token_id = kwargs.get("pad_token_id", 50256)
        
        # Initialize the model
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="./cache"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.model_type == "auto":
                # Try to determine model type automatically
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir="./cache",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    self.model_type = "causal"
                except:
                    try:
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.model_name,
                            cache_dir="./cache",
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            device_map="auto" if self.device == "cuda" else None
                        )
                        self.model_type = "seq2seq"
                    except Exception as e:
                        print(f"Could not load model automatically: {e}")
                        self._load_fallback_model()
            elif self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir="./cache",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir="./cache",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            # Move model to device if not using device_map
            if self.model is not None and self.device != "cuda":
                self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully. Type: {self.model_type}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a lightweight fallback model."""
        try:
            print("Loading fallback model: microsoft/DialoGPT-small")
            self.model_name = "microsoft/DialoGPT-small"
            self.model_type = "causal"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="./cache"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir="./cache"
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
                
            print("Fallback model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load fallback model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_available():
            return "Model not available. Please check the model configuration."
        
        try:
            # Override default parameters with kwargs
            max_length = kwargs.get("max_length", self.max_length)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            do_sample = kwargs.get("do_sample", self.do_sample)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            if self.device != "cuda":
                inputs = inputs.to(self.device)
            
            # Generate text
            with torch.no_grad():
                if self.model_type == "causal":
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                else:  # seq2seq
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        num_return_sequences=1
                    )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error during text generation: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_available():
            return {"error": "Model not available"}
        
        try:
            return {
                "model_name": self.model_name,
                "model_type": self.model_type,
                "device": self.device,
                "parameters": f"{self.model.num_parameters():,}",
                "max_length": self.max_length,
                "temperature": self.temperature
            }
        except Exception as e:
            return {"error": f"Could not get model info: {e}"}

class PipelineLLM(BaseLLM):
    """Interface for Hugging Face pipeline models."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 task: str = "text-generation",
                 **kwargs):
        """
        Initialize the pipeline LLM.
        
        Args:
            model_name: Name of the model
            task: Pipeline task type
            **kwargs: Additional pipeline parameters
        """
        self.model_name = model_name
        self.task = task
        self.pipeline = None
        
        # Pipeline parameters
        self.max_length = kwargs.get("max_length", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.do_sample = kwargs.get("do_sample", True)
        
        # Initialize the pipeline
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the pipeline."""
        try:
            print(f"Loading pipeline: {self.task} with {self.model_name}")
            
            self.pipeline = pipeline(
                self.task,
                model=self.model_name,
                cache_dir="./cache"
            )
            
            print("Pipeline loaded successfully")
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            self.pipeline = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the pipeline.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_available():
            return "Pipeline not available. Please check the configuration."
        
        try:
            # Override default parameters with kwargs
            max_length = kwargs.get("max_length", self.max_length)
            temperature = kwargs.get("temperature", self.temperature)
            do_sample = kwargs.get("do_sample", self.do_sample)
            
            # Generate text
            if self.task == "text-generation":
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.pipeline.tokenizer.pad_token_id,
                    num_return_sequences=1
                )
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0]["generated_text"]
                    # Remove the input prompt
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    return generated_text
                else:
                    return "No text generated"
            
            elif self.task == "summarization":
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    do_sample=do_sample
                )
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0]["summary_text"]
                else:
                    return "No summary generated"
            
            else:
                return f"Task {self.task} not supported"
                
        except Exception as e:
            print(f"Error during pipeline generation: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the pipeline is available."""
        return self.pipeline is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        if not self.is_available():
            return {"error": "Pipeline not available"}
        
        try:
            return {
                "model_name": self.model_name,
                "task": self.task,
                "max_length": self.max_length,
                "temperature": self.temperature
            }
        except Exception as e:
            return {"error": f"Could not get pipeline info: {e}"}

def create_llm(model_name: str = None, 
               model_type: str = "auto",
               use_pipeline: bool = False,
               **kwargs) -> BaseLLM:
    """
    Factory function to create an LLM instance.
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        use_pipeline: Whether to use pipeline approach
        **kwargs: Additional parameters
        
    Returns:
        LLM instance
    """
    if model_name is None:
        model_name = "microsoft/DialoGPT-medium"
    
    try:
        if use_pipeline:
            return PipelineLLM(model_name, **kwargs)
        else:
            return HuggingFaceLLM(model_name, model_type, **kwargs)
    except Exception as e:
        print(f"Error creating LLM: {e}")
        # Return a basic fallback
        return HuggingFaceLLM("microsoft/DialoGPT-small", **kwargs)

def get_available_models() -> List[str]:
    """Get a list of available model names."""
    return [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "gpt2",
        "gpt2-medium",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b"
    ]