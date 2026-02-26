"""GUI for Model Quantization using Gradio/Streamlit."""

import gradio as gr
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading

from ..local.quantization import ModelQuantizer, QuantizationConfig


class QuantizationGUI:
    """Web-based GUI for model quantization."""
    
    def __init__(self, quantizer: Optional[ModelQuantizer] = None):
        self.quantizer = quantizer or ModelQuantizer()
        self.current_job: Optional[threading.Thread] = None
        self.progress = 0
        self.status = "Ready"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        
        with gr.Blocks(title="NeuralNode Model Quantization") as demo:
            gr.Markdown("# 🧠 NeuralNode Model Quantization GUI")
            gr.Markdown("Convert and quantize LLM models to GGUF format")
            
            with gr.Tab("Quantize Model"):
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="Model Path",
                            placeholder="Path to HuggingFace model or GGUF",
                        )
                        
                        method = gr.Dropdown(
                            label="Quantization Method",
                            choices=list(self.quantizer.QUANTIZATION_METHODS.keys()),
                            value="Q4_K_M",
                        )
                        
                        output_path = gr.Textbox(
                            label="Output Path (optional)",
                            placeholder="Auto-generated if empty",
                        )
                    
                    with gr.Column():
                        vram_info = gr.Textbox(
                            label="GPU Info",
                            value=self._get_gpu_info(),
                            interactive=False,
                        )
                        
                        recommendation = gr.Textbox(
                            label="Recommended Method",
                            value=self._get_recommendation(),
                            interactive=False,
                        )
                
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress",
                    interactive=False,
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                )
                
                quantize_btn = gr.Button("🚀 Start Quantization", variant="primary")
                
                def start_quantization(model, method, output):
                    self.status = "Starting..."
                    self.progress = 0
                    
                    try:
                        result = self.quantizer.quantize(
                            model,
                            output_path=output if output else None,
                            method=method,
                        )
                        self.status = f"Completed: {result}"
                        self.progress = 100
                        return 100, self.status
                    except Exception as e:
                        self.status = f"Error: {str(e)}"
                        return 0, self.status
                
                quantize_btn.click(
                    fn=start_quantization,
                    inputs=[model_path, method, output_path],
                    outputs=[progress_bar, status_text],
                )
            
            with gr.Tab("Download Model"):
                model_id = gr.Textbox(
                    label="HuggingFace Model ID",
                    placeholder="e.g., meta-llama/Llama-2-7b-chat-hf",
                )
                
                download_btn = gr.Button("📥 Download Model")
                download_output = gr.Textbox(label="Download Status")
                
                def download_model(model_id):
                    try:
                        path = self.quantizer.download_model(model_id)
                        return f"Downloaded to: {path}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                download_btn.click(
                    fn=download_model,
                    inputs=[model_id],
                    outputs=[download_output],
                )
            
            with gr.Tab("Available Models"):
                models_list = gr.Dataframe(
                    headers=["Name", "Size (GB)", "Type", "License"],
                    value=self._get_models_list(),
                    interactive=False,
                )
            
            with gr.Tab("Local Models"):
                refresh_btn = gr.Button("🔄 Refresh")
                local_models = gr.Dataframe(
                    headers=["Name", "Format", "Size (GB)", "Path"],
                    interactive=False,
                )
                
                def refresh_local():
                    from ..local.quantization import ModelDownloader
                    downloader = ModelDownloader()
                    models = downloader.list_local_models()
                    return [[m["name"], m["format"], m["size_gb"], m["path"]] for m in models]
                
                refresh_btn.click(fn=refresh_local, outputs=[local_models])
        
        return demo
    
    def _get_gpu_info(self) -> str:
        """Get GPU information."""
        try:
            from ..local.hardware import HardwareDetector
            hw = HardwareDetector()
            if hw.has_cuda:
                return f"✅ CUDA Available\nVRAM: {hw.vram_gb:.1f} GB\nGPU Count: {torch.cuda.device_count()}"
            else:
                return "❌ No GPU available\nCPU-only mode"
        except Exception:
            return "Unable to detect GPU"
    
    def _get_recommendation(self) -> str:
        """Get quantization recommendation."""
        try:
            from ..local.hardware import HardwareDetector
            hw = HardwareDetector()
            method = self.quantizer.recommend_quantization(14.0, hw.vram_gb)  # 14B model
            return f"For 14B model with {hw.vram_gb:.1f}GB VRAM: {method}"
        except Exception:
            return "Q4_K_M (recommended)"
    
    def _get_models_list(self) -> List[List[str]]:
        """Get list of available models."""
        models = self.quantizer.list_available_models()
        return [[m["name"], str(m["size_gb"]), m["type"], m["license"]] for m in models[:20]]
    
    def launch(self, share: bool = False, port: int = 7860) -> None:
        """Launch the GUI."""
        demo = self.create_interface()
        demo.launch(share=share, server_port=port)


def launch_quantization_gui():
    """Quick launch function."""
    gui = QuantizationGUI()
    gui.launch()
