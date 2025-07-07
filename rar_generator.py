import json
import requests
import sglang as sgl
import re

class GeneratorMixin:
    def __init__(self, mode, server_url=None, model_name_or_path=None, server_params=None):
        """
        Define a GeneratorMixin class to generate RARs.

        Args:
            mode: The mode of the generator, can be "local" or "remote".
            server_url: The url of the server, only used when mode is "remote".
            model_name_or_path: The name or path of the model, only used when mode is "local".
            server_params: The parameters for the server, only used when mode is "remote".
        """
        # initialize the basic attributes that will be used in all cases
        self.server_url = server_url
        self.mode = mode
        self.model_name_or_path = model_name_or_path
        self.server_params = server_params or {}

        # use the post init method to set up the difference in the mode
        self._post_init()

    def _post_init(self):
        """
        Post-initialization method to set up any additional configuration
        This method can be overidden in subclasses for custom behavior
        """
        differences = {
            'offline': 'other',
            'localhost': '/generate',
            'proxy': '/llm'
        }
        mode_diff = differences.get(self.mode, False)
        if not mode_diff:
            raise ValueError(f"Invalid mode: {self.mode}. Supported modes are: {', '.join(differences.keys())}. Your mode is {self.mode}")
        else:
            if mode_diff == 'other':
                self._init_offline_model()
            else:
                self.server_url = self.server_url + mode_diff
                self._test_server_status()

    def _init_offline_model(self):
        """
        Initialize offline model with quantization support
        """
        # Create a copy of server_params to avoid modifying the original
        sgl_params = dict(self.server_params)
        
        # Check if we should use transformers fallback
        use_transformers_fallback = sgl_params.pop('use_transformers_fallback', False)
        quantization = sgl_params.get('quantization', None)
        
        # Map user-friendly quantization names to SGLang-compatible ones
        quantization_mapping = {
            'int8': 'blockwise_int8',  # Map int8 to blockwise_int8 for SGLang
            'int4': 'bitsandbytes',    # Map int4 to bitsandbytes for SGLang  
            'blockwise_int8': 'blockwise_int8',
            'bitsandbytes': 'bitsandbytes',
            'fp8': 'fp8',
            'w8a8_int8': 'w8a8_int8',
            'awq': 'awq',
            'gptq': 'gptq',
        }
        
        if use_transformers_fallback:
            print("Using transformers fallback as requested...")
            self._init_transformers_pipeline()
        else:
            # Map quantization to SGLang-compatible format
            if quantization and quantization in quantization_mapping:
                sgl_params['quantization'] = quantization_mapping[quantization]
                print(f"Mapping quantization '{quantization}' to '{quantization_mapping[quantization]}' for SGLang")
                
                # Add required parameters for specific quantization methods
                if quantization_mapping[quantization] == 'blockwise_int8':
                    if 'weight_block_size' not in sgl_params:
                        sgl_params['weight_block_size'] = 128  # Default block size
                        print(f"Setting default weight_block_size=128 for blockwise_int8")
                        
            elif quantization:
                print(f"Using quantization '{quantization}' directly for SGLang")
                
                # Add required parameters for blockwise_int8 if used directly
                if quantization == 'blockwise_int8' and 'weight_block_size' not in sgl_params:
                    sgl_params['weight_block_size'] = 128  # Default block size
                    print(f"Setting default weight_block_size=128 for blockwise_int8")
            
            try:
                # Set up SGLang engine with quantization support
                self.llm = sgl.Engine(
                    model_path=self.model_name_or_path,
                    **sgl_params
                )
                print(f"SGLang engine initialized successfully with quantization: {sgl_params.get('quantization', 'none')}")
            except Exception as e:
                print(f"SGLang initialization failed: {e}")
                print("Falling back to transformers with quantization...")
                self._init_transformers_pipeline()

    def _init_transformers_pipeline(self):
        """
        Initialize transformers pipeline with BitsAndBytes quantization
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            # Check if int8 quantization is requested
            quantization_type = self.server_params.get('quantization', None)
            
            if quantization_type == 'int8':
                # Configure BitsAndBytes for int8 quantization without CPU offload
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_enable_fp32_cpu_offload=False,  # Disable CPU offload
                )
                print("Loading model with int8 quantization and CPU offload disabled...")
            elif quantization_type == 'int4':
                # Configure BitsAndBytes for int4 quantization without CPU offload
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=False  # Disable CPU offload for int4 as well
                )
                print("Loading model with int4 quantization and CPU offload disabled...")
            else:
                bnb_config = None
                print("Loading model without quantization...")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device map based on quantization
            if quantization_type == 'int8':
                # Use more conservative device mapping for int8 with CPU offload
                device_map = "auto"
                max_memory = {0: "15GB"}  # Reserve some GPU memory
            else:
                device_map = "auto"
                max_memory = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.float16 if bnb_config is None else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
            )
            
            self.use_transformers = True
            print(f"Model loaded successfully with transformers. Memory usage reduced with quantization: {quantization_type}")
            
        except ImportError as e:
            print(f"BitsAndBytes not available: {e}")
            print("Please install bitsandbytes: pip install bitsandbytes")
            raise
        except Exception as e:
            print(f"Transformers initialization failed: {e}")
            raise

    def _test_server_status(self):
        """
        Test the status of the server
        This method can be overidden in subclasses for custom behavior
        Need to be implemented later
        """
        # TODO
        pass

    def generate(self, prompts, sampling_params={}):
        if hasattr(self, 'use_transformers') and self.use_transformers:
            return self._generate_transformers(prompts, sampling_params)
        elif self.mode == 'offline':
            return self._generate_offline(prompts, sampling_params)
        elif self.mode == 'localhost':
            return self._generate_localhost(prompts, sampling_params)
        elif self.mode == 'proxy':
            return self._generate_proxy(prompts, sampling_params)

    def _generate_transformers(self, prompts, sampling_params={}):
        """
        Generate responses using transformers with quantization
        """
        import torch
        from tqdm import tqdm
        
        outputs = []
        batch_size = 1  # Process one at a time to minimize memory usage
        print(f"🤖 Processing {len(prompts)} prompts with transformers + quantization (batch_size={batch_size})...")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
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
                max_length=1024  # Reduced max length to save memory
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(sampling_params.get('max_new_tokens', 512), 256),  # Limit new tokens
                    temperature=sampling_params.get('temperature', 0.6),
                    top_p=sampling_params.get('top_p', 0.95),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=sampling_params.get('repetition_penalty', 1.0),
                    use_cache=False  # Disable cache to save memory
                )
            
            # Decode batch results
            for i in range(len(batch_prompts)):
                # Decode only the new tokens
                input_length = len(inputs['input_ids'][i])
                new_tokens = generated_ids[i][input_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append({"text": generated_text})
            
            # Aggressive memory cleanup after each batch
            del inputs, generated_ids
            torch.cuda.empty_cache()
            
            # Force garbage collection every 10 batches
            if (batch_start + 1) % 10 == 0:
                import gc
                gc.collect()
        
        print(f"✅ Completed {len(outputs)} generations")
        return outputs
        
    def _generate_offline(self, prompts, sampling_params={}):
        """
        Generate responses for the given prompts using the specified model in offline mode.
        prompts: list of prompts to send to the LLMs
        sampling_params: Optional parameters for sampling
        return a JSON response from the LLM API
        """
        return self.llm.generate(prompts, sampling_params=sampling_params)
    
    def _generate_localhost(self, prompts, sampling_params={}):
        """
        Generate responses for the given prompts using the specified model in localhost mode.
        prompts: list of prompts to send to the LLMs
        sampling_params: Optional parameters for sampling
        return a JSON response from the LLM API
        """
        payload = {
            "text": prompts,
            "sampling_params": sampling_params
        }
        response = requests.post(self.server_url, json=payload)
        return response.json()
    
    def _generate_proxy(self, prompts, sampling_params={}):
        """
        Generate responses for the given prompts using the specified model in proxy mode.
        prompts: list of prompts to send to the LLMs
        sampling_params: Optional parameters for sampling
        return a JSON response from the LLM API
        """
        # resove openai and sglang parameter naming difference
        sampling_params['max_tokens'] = sampling_params.pop('max_new_tokens', None)
        sampling_params.pop("no_stop_trim", None)
        sampling_params.pop('repetition_penalty', None)

        payload = {
            "prompts": prompts,
            "model": self.model_name_or_path,
            "sampling_params": sampling_params
        }
        response = requests.post(self.server_url, json=payload)
        response = response.json()
        for out in response:
            out['text'] = out['text'].strip()
        return response
    
    def shutdown(self):
        """
        Shutdown the generator
        This method can be overidden in subclasses for custom behavior
        Need to be implemented later
        """
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
    
    def _fix_stop_word(self, text):
        """
        Fix missing stop words in text generated by the API

        When stop words are used as stopping criteria, the API often trims them from the repsponse. This function detects the most recently opend unclosed tag and add the stop word to the end of the text.
        (<search> or <answer>) and appends the corresponding closing tag.

        text: The generated text with potentially missing closing tags.
        return Text with appropriate clossing tag added if needed.
        """
        #Find last occurence of each opening tag
        last_search_open = text.rfind('<search>')
        last_answer_open = text.rfind('<answer>')

        # If neither tag is found, return the original text
        if last_search_open == -1 and last_answer_open == -1:
            return text
    
        # Determine which tag was opened most recently
        if last_search_open > last_answer_open:
            # check if the tag is already a closing tag after the last opening tag
            if text.find('</search>', last_search_open) == -1:
                # if not, add the closing tag
                text = text[:last_search_open] + '</search>' + text[last_search_open:]
        else:
            # check if the tag is already a closing tag after the last opening tag
            if text.find('</answer>', last_answer_open) == -1:
                # if not, add the closing tag
                text = text[:last_answer_open] + '</answer>' + text[last_answer_open:]
        return text