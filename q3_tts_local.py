# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "mlx-audio==0.3.0rc1",
#     "click",
#     "numpy",
#     "soundfile",
# ]
# ///
import os
import sys
import warnings
import logging
from pathlib import Path

# Silence non-critical warnings and logs for a cleaner terminal experience
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

import click
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


def get_unique_filename(base_path: Path) -> Path:
    """Return a unique filename, adding -2, -3, etc. if the file already exists."""
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    counter = 2
    while True:
        new_path = parent / f"{stem}-{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


class CustomHelpCommand(click.Command):
    def format_help(self, ctx, formatter):
        super().format_help(ctx, formatter)
        prog = ctx.info_name
        with formatter.section("Examples"):
            formatter.write_paragraph()
            formatter.write_text(f"{prog} \"say this text out loud\"")
            formatter.write_text(f"{prog} -o saved.wav \"hello world\"")
            formatter.write_text(f"{prog} -l Chinese \"你好世界\"")
            formatter.write_text(f"{prog} -i \"deep low voice\" \"hello\"")
            formatter.write_text(f"echo \"piped text\" | {prog}")


@click.command(cls=CustomHelpCommand)
@click.option("-o", "--output", default="output.wav", help="Output filename (default: output.wav)")
@click.option("-l", "--language", default="English", help="Language for TTS (default: English)")
@click.option("-i", "--instruct", default=None, help="Voice instruction (e.g., 'deep low voice')")
@click.option("-c", "--clone", default=None, help="Path to reference audio for cloning")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.argument("text", required=False)
def main(text: str | None, output: str, language: str, instruct: str | None, clone: str | None, verbose: bool):
    """Generate audio using Qwen3-TTS and MLX Audio."""
    # Handle piped input
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.UsageError("No text provided. Pass text as an argument or pipe it via stdin.")
    
    if not text:
        raise click.UsageError("Text cannot be empty.")
    
    # Determine output path
    output_path = Path(output)
    if output == "output.wav":
        # Only auto-increment for default filename
        output_path = get_unique_filename(output_path)
    
    # Determine model to load
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    if clone:
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
    if verbose:
        click.echo(f"Step 1: Loading model {model_id}...", err=True)
    
    model = load_model(model_id)
    
    if verbose:
        click.echo(f"Step 2: Model loaded successfully.", err=True)
        click.echo(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}", err=True)
    
    if verbose:
        click.echo(f"Step 3: Starting generation process...", err=True)
    
    if clone:
        # Build generation kwargs for cloning
        gen_kwargs = {
            "text": text,
            "ref_audio": clone,
            "language": language,
            "verbose": verbose,
            "x_vector_only_mode": True
        }
        # Generate with voice cloning
        results = list(model.generate_voice_clone(**gen_kwargs))
    else:
        # Build generation kwargs for voice design
        gen_kwargs = {
            "text": text,
            "language": language,
            "verbose": verbose,
            "instruct": instruct or "",
        }
        # Generate with voice design
        results = list(model.generate_voice_design(**gen_kwargs))
    
    if verbose:
        click.echo(f"Step 4: Generation complete. Saving audio...", err=True)
    
    audio = results[0].audio
    
    # Save to file
    sf.write(str(output_path), np.array(audio), model.sample_rate)
    if verbose:
        click.echo(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    main()
