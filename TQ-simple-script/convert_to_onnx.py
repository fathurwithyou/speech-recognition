import os
import torch
import torch.nn.functional as F
import whisper
from pathlib import Path
import onnx
from onnx import external_data_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
import tempfile


def patch_attention_for_onnx():
    """Patch Whisper attention to be ONNX-compatible."""
    
    def onnx_compatible_qkv_attention(self, q, k, v, mask=None):
        """ONNX-compatible version of qkv_attention."""
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            # Ensure mask slicing is compatible with ONNX
            if hasattr(mask, 'shape'):
                mask_slice = mask[:n_ctx, :n_ctx]
            else:
                mask_slice = mask
            qk = qk + mask_slice
        
        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
    
    def onnx_compatible_forward(self, x, xa=None, mask=None, kv_cache=None):
        """ONNX-compatible forward for MultiHeadAttention."""
        q = self.query(x)
        
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # cross-attention or no cache
            k = self.key(xa if xa is not None else x)
            v = self.value(xa if xa is not None else x)
        else:
            # cached self-attention
            k, v = kv_cache[self.key], kv_cache[self.value]
            
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
    
    # Patch the MultiHeadAttention class
    import whisper.model
    whisper.model.MultiHeadAttention.qkv_attention = onnx_compatible_qkv_attention
    whisper.model.MultiHeadAttention.forward = onnx_compatible_forward
    print("Patched Whisper attention for ONNX compatibility")


def export_large_model_to_onnx(module, dummy_input, output_path, input_names, output_names, dynamic_axes):
    """Export large models to ONNX with external data support."""
    print(f"Exporting large model to {output_path}...")
    
    # Use external data by default for large models
    temp_path = str(output_path).replace('.onnx', '_temp.onnx')
    
    try:
        # Export with minimal settings to avoid 2GB issue
        torch.onnx.export(
            module,
            dummy_input,
            temp_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=False,  # Disable to reduce model size during export
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False
        )
        
        # Load the model and save with external data
        model = onnx.load(temp_path)
        
        # Save with external data
        data_filename = os.path.basename(str(output_path)).replace('.onnx', '.data')
        onnx.save_model(
            model, 
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024
        )
        
        print(f"Model exported with external data file: {data_filename}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        raise e
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def export_whisper_to_onnx(model_name="base", output_dir="onnx_models"):
    """Convert Whisper model to ONNX format."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Patch attention for ONNX compatibility
    patch_attention_for_onnx()
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    model.eval()
    
    # Create dummy inputs for export
    # Whisper expects mel spectrogram input
    mel_shape = (1, model.dims.n_mels, 3000)  # batch_size, n_mels, time_steps
    dummy_mel = torch.randn(mel_shape)
    
    # For tokens (decoder input)
    dummy_tokens = torch.tensor([[50258]], dtype=torch.long)  # Start token
    
    print("Exporting encoder to ONNX...")
    encoder_path = output_path / f"whisper_{model_name}_encoder.onnx"
    
    # Export encoder with large model support
    export_large_model_to_onnx(
        model.encoder,
        dummy_mel,
        encoder_path,
        input_names=["mel_spectrogram"],
        output_names=["encoder_output"],
        dynamic_axes={
            "mel_spectrogram": {2: "time_steps"},
            "encoder_output": {1: "time_steps"}
        }
    )
    
    print("Exporting decoder to ONNX...")
    decoder_path = output_path / f"whisper_{model_name}_decoder.onnx"
    
    # Create dummy encoder output for decoder export
    encoder_output_shape = (1, 1500, model.dims.n_audio_state)
    dummy_encoder_output = torch.randn(encoder_output_shape)
    
    # Export decoder with large model support
    export_large_model_to_onnx(
        model.decoder,
        (dummy_tokens, dummy_encoder_output),
        decoder_path,
        input_names=["tokens", "encoder_output"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {1: "token_length"},
            "encoder_output": {1: "time_steps"},
            "logits": {1: "token_length"}
        }
    )
    
    print(f"ONNX models exported to {output_path}")
    return encoder_path, decoder_path


def quantize_onnx_model(model_path, output_path=None):
    """Apply INT8 quantization to ONNX model."""
    
    if output_path is None:
        # Create quantized version name
        path = Path(model_path)
        output_path = path.parent / f"{path.stem}_int8{path.suffix}"
    
    print(f"Quantizing {model_path} to INT8...")
    
    try:
        # Check if external data file exists (simpler detection)
        model_dir = Path(model_path).parent
        data_file = model_dir / (Path(model_path).stem + ".data")
        
        if data_file.exists():
            print("Model has external data file, skipping quantization (may cause corruption)")
            print("FP32 ONNX is still much faster than PyTorch models")
            return None
            
        # Also check for large model names that typically have external data
        if any(size in str(model_path).lower() for size in ['large', 'medium']):
            print("Large model detected, skipping quantization to avoid corruption")
            print("FP32 ONNX is still much faster than PyTorch models")
            return None
        
        # Standard quantization for smaller models without external data
        print("Proceeding with quantization for smaller model...")
        try:
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
        except TypeError:
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8
            )
        
        # Verify the quantized model can be loaded
        try:
            test_model = onnx.load(str(output_path))
            onnx.checker.check_model(test_model)
            print(f"✅ Quantized model saved and verified: {output_path}")
            return output_path
        except Exception as verify_error:
            print(f"❌ Quantized model verification failed: {verify_error}")
            # Clean up corrupted file
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None
            
    except Exception as e:
        print(f"Quantization failed: {e}")
        return None


def convert_whisper_to_onnx_int8(model_name="base", output_dir="onnx_models"):
    """Complete pipeline: Convert Whisper to ONNX and quantize to INT8."""
    
    print(f"Converting Whisper {model_name} to ONNX INT8...")
    
    # Step 1: Export to ONNX
    encoder_path, decoder_path = export_whisper_to_onnx(model_name, output_dir)
    
    # Step 2: Quantize to INT8
    encoder_int8_path = quantize_onnx_model(encoder_path)
    decoder_int8_path = quantize_onnx_model(decoder_path)
    
    # Check results and provide fallback
    success_count = 0
    if encoder_int8_path:
        print(f"✅ Encoder quantized: {encoder_int8_path}")
        success_count += 1
    else:
        print(f"⚠️  Encoder quantization failed, using FP32: {encoder_path}")
        encoder_int8_path = encoder_path
    
    if decoder_int8_path:
        print(f"✅ Decoder quantized: {decoder_int8_path}")
        success_count += 1
    else:
        print(f"⚠️  Decoder quantization failed, using FP32: {decoder_path}")
        decoder_int8_path = decoder_path
    
    if success_count > 0:
        print(f"✅ Conversion completed with {success_count}/2 models quantized")
        print(f"Encoder: {encoder_int8_path}")
        print(f"Decoder: {decoder_int8_path}")
        return encoder_int8_path, decoder_int8_path
    else:
        print("⚠️  No models quantized, but FP32 models available")
        return encoder_path, decoder_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Whisper model to ONNX INT8")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--output-dir", default="onnx_models", help="Output directory for ONNX models")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert model
    encoder_path, decoder_path = convert_whisper_to_onnx_int8(args.model, args.output_dir)
    
    if encoder_path and decoder_path:
        # Verify the models
        try:
            import onnxruntime as ort
            
            print("\nVerifying ONNX models...")
            
            # Test encoder
            encoder_session = ort.InferenceSession(str(encoder_path))
            print(f"✅ Encoder model loaded successfully")
            print(f"   Input: {encoder_session.get_inputs()[0].name} {encoder_session.get_inputs()[0].shape}")
            print(f"   Output: {encoder_session.get_outputs()[0].name} {encoder_session.get_outputs()[0].shape}")
            
            # Test decoder  
            decoder_session = ort.InferenceSession(str(decoder_path))
            print(f"✅ Decoder model loaded successfully")
            print(f"   Inputs: {[inp.name for inp in decoder_session.get_inputs()]}")
            print(f"   Output: {decoder_session.get_outputs()[0].name} {decoder_session.get_outputs()[0].shape}")
            
        except Exception as e:
            print(f"❌ Model verification failed: {e}")