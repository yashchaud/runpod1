"""
vLLM client for Qwen3-VL inference with video/image inputs.
Handles both direct vLLM usage and OpenAI-compatible API calls.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import base64
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for interacting with Qwen3-VL via vLLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct-FP8",
        max_model_len: int = 131072,  # 128K tokens
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize vLLM client.

        Args:
            model_name: HuggingFace model identifier
            max_model_len: Maximum context length
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.tokenizer = None

    def initialize_model(self) -> None:
        """Initialize vLLM model (lazy loading)."""
        if self.llm is not None:
            logger.info("Model already initialized")
            return

        try:
            from vllm import LLM

            logger.info(f"Initializing vLLM with model: {self.model_name}")

            self.llm = LLM(
                model=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                # Multimodal optimizations
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            )

            logger.info("vLLM model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            raise

    def create_video_message(
        self,
        video_path: Optional[str] = None,
        frame_paths: Optional[List[str]] = None,
        prompt: str = "Describe this video in detail.",
    ) -> List[Dict[str, Any]]:
        """
        Create message format for video/image input.

        Args:
            video_path: Path to video file (for direct video input)
            frame_paths: List of image frame paths (alternative to video_path)
            prompt: Text prompt for the model

        Returns:
            Messages list in Qwen3-VL format
        """
        content = []

        # Add video or images
        if video_path:
            content.append({
                "type": "video",
                "video": f"file://{os.path.abspath(video_path)}"
            })
        elif frame_paths:
            # Add frames as individual images
            for frame_path in frame_paths:
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(frame_path)}"
                })
        else:
            raise ValueError("Either video_path or frame_paths must be provided")

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{
            "role": "user",
            "content": content
        }]

        return messages

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        **kwargs
    ) -> str:
        """
        Generate response from vLLM model.

        Args:
            messages: Messages in chat format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated text response
        """
        if self.llm is None:
            self.initialize_model()

        try:
            from vllm import SamplingParams

            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )

            # Format messages for vLLM
            # For Qwen3-VL, messages are passed directly
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=sampling_params,
                use_tqdm=False
            )

            if not outputs or len(outputs) == 0:
                raise RuntimeError("No output generated")

            # Extract generated text
            response = outputs[0].outputs[0].text

            logger.info(f"Generated response ({len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def process_video_chunk(
        self,
        video_path: Optional[str] = None,
        frame_paths: Optional[List[str]] = None,
        prompt: str = "Describe what happens in this video segment.",
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single video chunk.

        Args:
            video_path: Path to video chunk
            frame_paths: Alternative: list of frame images
            prompt: Analysis prompt
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        try:
            # Create messages
            messages = self.create_video_message(
                video_path=video_path,
                frame_paths=frame_paths,
                prompt=prompt
            )

            # Generate response
            response = self.generate(
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "response": response,
                "processing_time": processing_time,
                "error": None
            }

        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return {
                "success": False,
                "response": None,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

    def process_chunks_parallel(
        self,
        chunks: List[Dict[str, Any]],
        prompt: str = "Describe what happens in this video segment.",
        max_workers: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple video chunks in parallel.

        Args:
            chunks: List of chunk dictionaries with 'video_path' or 'frame_paths'
            prompt: Analysis prompt
            max_workers: Maximum parallel workers

        Returns:
            List of results for each chunk
        """
        logger.info(f"Processing {len(chunks)} chunks with {max_workers} workers")

        # For vLLM, we can use continuous batching by making sequential calls
        # vLLM handles batching internally
        results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

            result = self.process_video_chunk(
                video_path=chunk.get("video_path"),
                frame_paths=chunk.get("frame_paths"),
                prompt=prompt,
                **kwargs
            )

            result["chunk_index"] = chunk.get("chunk_index", i)
            result["start_time"] = chunk.get("start_time", 0)
            result["end_time"] = chunk.get("end_time", 0)

            results.append(result)

        successful = sum(1 for r in results if r["success"])
        logger.info(
            f"Completed {len(results)} chunks: "
            f"{successful} successful, {len(results) - successful} failed"
        )

        return results

    async def process_chunks_async(
        self,
        chunks: List[Dict[str, Any]],
        prompt: str = "Describe what happens in this video segment.",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process chunks asynchronously (for use with async frameworks).

        Args:
            chunks: List of chunk dictionaries
            prompt: Analysis prompt

        Returns:
            List of results for each chunk
        """
        loop = asyncio.get_event_loop()
        results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (async)")

            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None,
                self.process_video_chunk,
                chunk.get("video_path"),
                chunk.get("frame_paths"),
                prompt,
                kwargs.get("max_tokens", 512)
            )

            result["chunk_index"] = chunk.get("chunk_index", i)
            results.append(result)

        return results

    def aggregate_chunk_results(
        self,
        results: List[Dict[str, Any]],
        aggregation_prompt: Optional[str] = None
    ) -> str:
        """
        Aggregate results from multiple chunks into coherent summary.

        Args:
            results: List of chunk processing results
            aggregation_prompt: Optional custom prompt for aggregation

        Returns:
            Aggregated summary
        """
        # Filter successful results
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            return "No successful chunk results to aggregate."

        # Sort by chunk index
        successful_results.sort(key=lambda x: x.get("chunk_index", 0))

        # Create aggregation content
        chunk_summaries = []
        for i, result in enumerate(successful_results):
            start = result.get("start_time", 0)
            end = result.get("end_time", 0)
            response = result.get("response", "")

            chunk_summaries.append(
                f"Segment {i + 1} ({start:.1f}s - {end:.1f}s):\n{response}"
            )

        combined_text = "\n\n".join(chunk_summaries)

        # If aggregation prompt provided, use model to synthesize
        if aggregation_prompt:
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{aggregation_prompt}\n\n{combined_text}"}
                    ]
                }]

                return self.generate(messages, max_tokens=1024)

            except Exception as e:
                logger.error(f"Aggregation with model failed: {e}")
                # Fallback to simple concatenation
                return combined_text

        return combined_text
