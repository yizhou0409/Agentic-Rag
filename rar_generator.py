import json
import requests
import sglang as sgl
import re
import os
import logging
from typing import List, Dict, Any, Optional, Union, cast
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for model generation parameters."""
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.0

class GeneratorMixin:
    """
    A mixin class for generating RAR (Reasoning and Retrieval) responses.
    Supports multiple modes: offline (local), localhost, proxy, and OpenAI.
    """
    
    def __init__(
        self, 
        mode: str, 
        server_url: Optional[str] = None, 
        model_name_or_path: Optional[str] = None, 
        server_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GeneratorMixin.
        
        Args:
            mode: The generation mode ('offline', 'localhost', 'proxy', 'openai').
            server_url: The server URL for remote modes.
            model_name_or_path: The model path or name.
            server_params: Additional server parameters.
            
        Raises:
            ValueError: If mode is not supported.
            ImportError: If required packages are not available.
        """
        self.server_url = server_url
        self.mode = mode
        self.model_name_or_path = model_name_or_path
        self.server_params = server_params or {}
        self._post_init()

    def _post_init(self) -> None:
        """Post-initialization setup based on mode."""
        mode_configs = {
            'offline': 'other',
            'localhost': '/generate',
            'proxy': '/llm',
            'openai': 'openai'
        }
        
        mode_diff = mode_configs.get(self.mode)
        if not mode_diff:
            supported_modes = ', '.join(mode_configs.keys())
            raise ValueError(f"Invalid mode: {self.mode}. Supported modes: {supported_modes}")
        
        if mode_diff == 'other':
            self._init_offline_model()
        elif mode_diff == 'openai':
            if openai is None:
                raise ImportError("openai package required for openai mode. Install with 'pip install openai'.")
        else:
            if self.server_url:
                self.server_url = self.server_url + mode_diff
            self._test_server_status()

    def _init_offline_model(self) -> None:
        """Initialize offline model with quantization support."""
        sgl_params = dict(self.server_params)
        use_transformers_fallback = sgl_params.pop('use_transformers_fallback', False)
        quantization = sgl_params.get('quantization')
        
        quantization_mapping = {
            'int8': 'blockwise_int8',
            'int4': 'bitsandbytes',
            'blockwise_int8': 'blockwise_int8',
            'bitsandbytes': 'bitsandbytes',
            'fp8': 'fp8',
            'w8a8_int8': 'w8a8_int8',
            'awq': 'awq',
            'gptq': 'gptq',
        }
        
        if use_transformers_fallback:
            logger.info("Using transformers fallback as requested...")
            self._init_transformers_pipeline()
            return
        
        # Map quantization to SGLang-compatible format
        if quantization and quantization in quantization_mapping:
            sgl_params['quantization'] = quantization_mapping[quantization]
            logger.info(f"Mapping quantization '{quantization}' to '{quantization_mapping[quantization]}' for SGLang")
            
            # Add required parameters for specific quantization methods
            if quantization_mapping[quantization] == 'blockwise_int8':
                if 'weight_block_size' not in sgl_params:
                    sgl_params['weight_block_size'] = 128
                    logger.info("Setting default weight_block_size=128 for blockwise_int8")
        elif quantization:
            logger.info(f"Using quantization '{quantization}' directly for SGLang")
            if quantization == 'blockwise_int8' and 'weight_block_size' not in sgl_params:
                sgl_params['weight_block_size'] = 128
                logger.info("Setting default weight_block_size=128 for blockwise_int8")
        
        try:
            self.llm = sgl.Engine(model_path=self.model_name_or_path, **sgl_params)
            logger.info(f"SGLang engine initialized successfully with quantization: {sgl_params.get('quantization', 'none')}")
        except Exception as e:
            logger.warning(f"SGLang initialization failed: {e}")
            logger.info("Falling back to transformers with quantization...")
            self._init_transformers_pipeline()

    def _init_transformers_pipeline(self) -> None:
        """Initialize transformers pipeline with BitsAndBytes quantization or multi-GPU support."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            quantization_type = self.server_params.get('quantization')
            llm_int8_enable_fp32_cpu_offload = self.server_params.get('llm_int8_enable_fp32_cpu_offload', False)
            use_multi_gpu = self.server_params.get('use_multi_gpu', True)  # Default to multi-GPU
            num_gpus = self.server_params.get('num_gpus', torch.cuda.device_count())
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Multi-GPU setup without quantization (default behavior)
            if use_multi_gpu and not quantization_type:
                logger.info(f"Loading model on {num_gpus} GPUs without quantization...")
                
                # Set up device map for multi-GPU
                if num_gpus > 1:
                    # Use auto device map for multi-GPU
                    device_map = "auto"
                    max_memory = {}
                    for i in range(num_gpus):
                        max_memory[i] = f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 * 0.9)}GB"
                    max_memory["cpu"] = "100GB"  # Allow some CPU memory for offloading
                else:
                    device_map = {"": 0}
                    max_memory = {0: "75GB"}
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                logger.info(f"Model loaded successfully on {num_gpus} GPUs without quantization")
                
            # Single GPU with quantization
            else:
                if quantization_type == 'int8':
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
                    )
                    logger.info(f"Loading model with int8 quantization and CPU offload {'enabled' if llm_int8_enable_fp32_cpu_offload else 'disabled'}...")
                elif quantization_type == 'int4':
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
                    )
                    logger.info(f"Loading model with int4 quantization and CPU offload {'enabled' if llm_int8_enable_fp32_cpu_offload else 'disabled'}...")
                else:
                    bnb_config = None
                    logger.info("Loading model without quantization on single GPU...")

                # Determine device map based on quantization
                device_map = {"": 0}  # Force everything on GPU 0
                max_memory = {0: "75GB"} if quantization_type == 'int8' else {0: "75GB"}
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    max_memory=max_memory,
                    torch_dtype=torch.float16 if bnb_config is None else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                logger.info(f"Model loaded successfully with transformers. Memory usage reduced with quantization: {quantization_type}")
            
            self.use_transformers = True
            
        except ImportError as e:
            logger.error(f"BitsAndBytes not available: {e}")
            logger.error("Please install bitsandbytes: pip install bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"Transformers initialization failed: {e}")
            raise

    def _test_server_status(self) -> None:
        """Test the status of the server. To be implemented in subclasses."""
        # TODO: Implement server status testing
        pass

    def generate(
        self, 
        prompts: Union[List[str], List[Dict[str, Any]]], 
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts (strings or message dicts).
            sampling_params: Optional generation parameters.
            
        Returns:
            List of response dicts with 'text' key.
        """
        sampling_params = sampling_params or {}
        
        if hasattr(self, 'use_transformers') and self.use_transformers:
            return self._generate_transformers(cast(List[str], prompts), sampling_params)
        elif self.mode == 'offline':
            return self._generate_offline(cast(List[str], prompts), sampling_params)
        elif self.mode == 'localhost':
            return self._generate_localhost(cast(List[str], prompts), sampling_params)
        elif self.mode == 'proxy':
            return self._generate_proxy(cast(List[str], prompts), sampling_params)
        elif self.mode == 'openai':
            return self._generate_openai(cast(List[Dict[str, Any]], prompts), sampling_params)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _generate_transformers(
        self, 
        prompts: List[str], 
        sampling_params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate responses using transformers with quantization or multi-GPU."""
        import torch
        from tqdm import tqdm
        
        use_multi_gpu = self.server_params.get('use_multi_gpu', True)  # Default to multi-GPU
        
        # Determine batch size based on setup
        if use_multi_gpu:
            batch_size = min(4, len(prompts))  # Larger batch size for multi-GPU
            logger.info(f"Processing {len(prompts)} prompts with transformers on multiple GPUs (batch_size={batch_size})...")
        else:
            batch_size = 1  # Process one at a time to minimize memory usage for single GPU
            logger.info(f"Processing {len(prompts)} prompts with transformers + quantization (batch_size={batch_size})...")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
        outputs = []
        
        # Process in batches
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Processing batches", ncols=100):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            
            # Move inputs to the appropriate device(s)
            if use_multi_gpu:
                # For multi-GPU, inputs will be automatically distributed
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(sampling_params.get('max_new_tokens', 512), 256),
                    temperature=sampling_params.get('temperature', 0.6),
                    top_p=sampling_params.get('top_p', 0.95),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=sampling_params.get('repetition_penalty', 1.0),
                    use_cache=False
                )
            
            # Decode batch results
            for i in range(len(batch_prompts)):
                input_length = len(inputs['input_ids'][i])
                new_tokens = generated_ids[i][input_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append({"text": generated_text})
            
            # Memory cleanup
            del inputs, generated_ids
            torch.cuda.empty_cache()
            
            # Force garbage collection every 10 batches
            if (batch_start + 1) % 10 == 0:
                import gc
                gc.collect()
        
        logger.info(f"Completed {len(outputs)} generations")
        return outputs
        
    def _generate_offline(
        self, 
        prompts: List[str], 
        sampling_params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate responses using SGLang engine in offline mode."""
        return self.llm.generate(prompts, sampling_params=sampling_params)
    
    def _generate_localhost(
        self, 
        prompts: List[str], 
        sampling_params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate responses using localhost server."""
        if not self.server_url:
            raise ValueError("server_url is required for localhost mode")
        
        payload = {
            "text": prompts,
            "sampling_params": sampling_params
        }
        response = requests.post(self.server_url, json=payload)
        return response.json()
    
    def _generate_proxy(
        self, 
        prompts: List[str], 
        sampling_params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate responses using proxy server."""
        if not self.server_url:
            raise ValueError("server_url is required for proxy mode")
        
        # Resolve OpenAI and SGLang parameter naming differences
        proxy_params = dict(sampling_params)
        proxy_params['max_tokens'] = proxy_params.pop('max_new_tokens', None)
        proxy_params.pop("no_stop_trim", None)
        proxy_params.pop('repetition_penalty', None)

        payload = {
            "prompts": prompts,
            "model": self.model_name_or_path,
            "sampling_params": proxy_params
        }
        response = requests.post(self.server_url, json=payload)
        response_data = response.json()
        for out in response_data:
            out['text'] = out['text'].strip()
        return response_data
    
    def _generate_openai(
        self, 
        prompts: List[Dict[str, Any]], 
        sampling_params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate responses using OpenAI API."""
        api_key = os.environ.get("OPENAI_API_KEY", self.server_params.get("api_key"))
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key in server_params must be set for openai mode.")
        
        import openai as _openai
        _openai.api_key = api_key
        model = self.model_name_or_path or self.server_params.get("model", "gpt-4o")
        results = []
        
        for messages in prompts:
            # Ensure messages is a list of dicts for OpenAI API
            if isinstance(messages, dict):
                messages = [messages]
            
            # Map max_new_tokens to max_tokens for OpenAI API
            params = {k: v for k, v in sampling_params.items() if v is not None}
            if "max_new_tokens" in params:
                params["max_tokens"] = params.pop("max_new_tokens")
            
            # Remove unsupported keys
            for unsupported in ["no_stop_trim", "repetition_penalty", "stop"]:
                params.pop(unsupported, None)
            
            response = _openai.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            text = response.choices[0].message.content
            results.append({"text": text})
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown the generator and clean up resources."""
        if self.mode == 'offline':
            if hasattr(self, 'use_transformers') and self.use_transformers:
                # Clean up transformers model
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                import torch
                torch.cuda.empty_cache()
            elif hasattr(self, 'llm'):
                self.llm.shutdown()
    
    def _fix_stop_word(self, text: str) -> str:
        """
        Fix missing stop words in text generated by the API.
        
        When stop words are used as stopping criteria, the API often trims them from the response.
        This function detects the most recently opened unclosed tag and adds the stop word to the end.
        
        Args:
            text: The generated text with potentially missing closing tags.
            
        Returns:
            Text with appropriate closing tag added if needed.
        """
        # Find last occurrence of each opening tag
        last_search_open = text.rfind('<search>')
        last_answer_open = text.rfind('<answer>')

        # If neither tag is found, return the original text
        if last_search_open == -1 and last_answer_open == -1:
            return text
    
        # Determine which tag was opened most recently
        if last_search_open > last_answer_open:
            # Check if the tag is already closed after the last opening tag
            if text.find('</search>', last_search_open) == -1:
                # If not, add the closing tag
                text = text[:last_search_open] + '</search>' + text[last_search_open:]
        else:
            # Check if the tag is already closed after the last opening tag
            if text.find('</answer>', last_answer_open) == -1:
                # If not, add the closing tag
                text = text[:last_answer_open] + '</answer>' + text[last_answer_open:]
        
        return text