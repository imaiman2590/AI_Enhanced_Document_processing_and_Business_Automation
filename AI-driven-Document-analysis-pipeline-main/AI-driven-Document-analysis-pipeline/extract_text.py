from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import io


@dataclass
class ExtractionConfig:
    max_length: int = 512
    dpi: int = 200
    early_stopping: bool = True
    temperature: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    task_prompt: str = "<s>"
    batch_size: int = 1
    max_workers: int = 4
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    low_cpu_mem_usage: bool = True
    use_cache: bool = True


class TextExtractorEngine:
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        config: Optional[ExtractionConfig] = None,
        enable_logging: bool = True,
        log_level: str = "INFO"
    ):
        self.model_name = model_name
        self.config = config or ExtractionConfig()
        self.logger = self._setup_logging(enable_logging, log_level)
        
        # Initialize device
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self._load_model()
        
    def _setup_logging(self, enable: bool, level: str) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if enable:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, level.upper()))
        return logger
    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device:
            return torch.device(self.config.device)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            
            # Load model with configuration
            model_kwargs = {
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
                "use_cache": self.config.use_cache
            }
            
            if self.config.torch_dtype:
                model_kwargs["torch_dtype"] = self.config.torch_dtype
            
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, **model_kwargs
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")
    
    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        page_range: Optional[tuple] = None,
        parallel: bool = True
    ) -> Union[str, List[str]]:
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            self.logger.info(f"Processing PDF: {pdf_path}")
            
            # Convert PDF to images
            convert_kwargs = {"dpi": self.config.dpi}
            if page_range:
                convert_kwargs.update({
                    "first_page": page_range[0],
                    "last_page": page_range[1]
                })
            
            images = convert_from_path(str(pdf_path), **convert_kwargs)
            self.logger.info(f"Converted {len(images)} pages to images")
            
            # Process images
            if parallel and len(images) > 1:
                results = self._process_images_parallel(images)
            else:
                results = [self._process_image_with_donut(img) for img in images]
            
            # Return results
            if len(results) == 1:
                return results[0]
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
    
    def extract_from_image(
        self,
        image_path: Union[str, Path, Image.Image, io.BytesIO]
    ) -> str:
        try:
            # Handle different input types
            if isinstance(image_path, Image.Image):
                image = image_path
            elif isinstance(image_path, io.BytesIO):
                image = Image.open(image_path)
            else:
                image_path = Path(image_path)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            # Ensure RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            self.logger.info(f"Processing image: {image.size}")
            return self._process_image_with_donut(image)
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise
    
    def extract_from_batch(
        self,
        paths: List[Union[str, Path]],
        parallel: bool = True
    ) -> Dict[str, str]:
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_path = {}
                
                for path in paths:
                    path_obj = Path(path)
                    if path_obj.suffix.lower() == '.pdf':
                        future = executor.submit(self.extract_from_pdf, path, None, False)
                    else:
                        future = executor.submit(self.extract_from_image, path)
                    future_to_path[future] = str(path)
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing {path}: {e}")
                        results[path] = ""
        else:
            for path in paths:
                try:
                    path_obj = Path(path)
                    if path_obj.suffix.lower() == '.pdf':
                        results[str(path)] = self.extract_from_pdf(path, None, False)
                    else:
                        results[str(path)] = self.extract_from_image(path)
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {e}")
                    results[str(path)] = ""
        
        return results
    
    def _process_images_parallel(self, images: List[Image.Image]) -> List[str]:
        results = [""] * len(images)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_image_with_donut, img): i
                for i, img in enumerate(images)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing image {index}: {e}")
                    results[index] = ""
        
        return results
    
    def _process_image_with_donut(self, image: Image.Image) -> str:
        try:
            # Prepare input
            pixel_values = self.processor(
                image, return_tensors="pt"
            ).pixel_values.to(self.device)
            
            decoder_input_ids = self.processor.tokenizer(
                self.config.task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Generate text
            generation_kwargs = {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
                "max_length": self.config.max_length,
                "early_stopping": self.config.early_stopping,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "temperature": self.config.temperature,
                "do_sample": self.config.do_sample,
                "num_beams": self.config.num_beams,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode output
            decoded_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            
            return decoded_text
            
        except Exception as e:
            self.logger.error(f"Error in Donut processing: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "config": self.config.__dict__,
            "processor_vocab_size": len(self.processor.tokenizer) if self.processor else None,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else None
        }
    
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'processor') and self.processor:
            del self.processor
        torch.cuda.empty_cache()


# Using way:
if __name__ == "__main__":
    extractor = TextExtractorEngine()
    config = ExtractionConfig(
        max_length=1024,
        dpi=300,
        temperature=0.7,
        max_workers=8
    )
    
    # Advanced usage with custom model
    advanced_extractor = TextExtractorEngine(
        model_name="naver-clova-ix/donut-base-finetuned-docvqa",
        config=config,
        enable_logging=True,
        log_level="DEBUG"
    )
    
    # Extract from PDF
      pdf_text = extractor.extract_from_pdf("document.pdf")
    
    # Extract from image
      image_text = extractor.extract_from_image("image.jpg")
    
    # Batch processing
      batch_results = extractor.extract_from_batch(["file1.pdf", "file2.jpg"])
    
    # Update configuration on the fly
      extractor.update_config(max_length=2048, temperature=0.5)
    
    print("Text Extractor Engine initialized successfully!")
