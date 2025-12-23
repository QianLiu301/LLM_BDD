"""
LLM BDD Generator - Flask Backend
==================================

Simple backend for generating BDD feature files using LLMs.

Run: python main.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# Proxy Setup
# ============================================================================
def setup_proxy():
    """Setup proxy from config file for international APIs"""
    config_paths = [
        Path('config/llm_config.json'),
        Path('llm_config.json'),
        Path('../config/llm_config.json'),
        Path(__file__).parent / 'config' / 'llm_config.json',
        Path(__file__).parent.parent / 'config' / 'llm_config.json',
    ]

    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        proxy_config = config.get('proxy', {})
        if proxy_config.get('enabled'):
            http_proxy = proxy_config.get('http_proxy', '')
            https_proxy = proxy_config.get('https_proxy', '')
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
            os.environ['http_proxy'] = http_proxy
            os.environ['https_proxy'] = https_proxy
            print(f"🌐 Proxy enabled: {https_proxy}")
    except Exception as e:
        print(f"⚠️  Proxy config error: {e}")

setup_proxy()

# ============================================================================
# Import Modules
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from feature_generator_llm import FeatureGeneratorLLM
    from llm_providers import LLMFactory
    HAS_MODULES = True
    print("✅ Modules imported successfully")

except ImportError as e:
    print(f"⚠️  Warning: Could not import modules: {e}")
    HAS_MODULES = False

# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Store last generated file info
last_generated = {
    'filename': None,
    'filepath': None,
    'llm': None
}


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'bdd_generator.html')


@app.route('/api/generate', methods=['POST'])
def generate_bdd():
    """
    Generate BDD Feature file using LLM

    Request:
    {
        "llm": "groq|deepseek|openai|claude|gemini",
        "model": "gpt-5-mini",  // only for openai
        "input": "16-bit ALU with ADD, SUB"
    }

    Response:
    {
        "success": true,
        "filename": "alu_16bit_xxx.feature",
        "preview": "Feature: ...",
        "llm": "groq"
    }
    """
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported. Check if feature_generator_llm.py exists.'
        }), 500

    try:
        data = request.json
        llm_name = data.get('llm', 'groq')
        model = data.get('model')
        user_input = data.get('input', '')

        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Please enter your requirements'
            }), 400

        print(f"\n{'='*60}")
        print(f"🚀 Generating BDD Feature")
        print(f"{'='*60}")
        print(f"   LLM: {llm_name.upper()}")
        if model:
            print(f"   Model: {model}")
        print(f"   Input: {user_input[:50]}...")
        print()

        # Create generator - let it create its own LLM
        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            debug=True
        )

        # For OpenAI with specific model, we need to override
        if model and llm_name == 'openai':
            try:
                llm = LLMFactory.create_provider('openai', model=model)
                generator.llm = llm
                print(f"   ✅ OpenAI model set to: {model}")
            except Exception as e:
                print(f"   ⚠️ Failed to set model {model}: {e}")

        # Generate feature
        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({
                'success': False,
                'error': 'LLM returned empty response. Please try again.'
            }), 500

        # Verify file
        feature_path_obj = Path(feature_path)
        if not feature_path_obj.exists():
            return jsonify({
                'success': False,
                'error': f'File was not created: {feature_path}'
            }), 500

        # Read content for preview
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Store last generated info
        last_generated['filename'] = feature_path_obj.name
        last_generated['filepath'] = str(feature_path)
        last_generated['llm'] = llm_name

        print(f"\n{'='*60}")
        print(f"✅ Success! File: {feature_path_obj.name}")
        print(f"{'='*60}\n")

        return jsonify({
            'success': True,
            'filename': feature_path_obj.name,
            'preview': content[:500] + ('...' if len(content) > 500 else ''),
            'full_content': content,
            'llm': llm_name,
            'model': model or llm_name,
            'filepath': str(feature_path)
        })

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """
    Download generated BDD file

    Searches in output/bdd/{llm}/ directories
    """
    print(f"\n📥 Download request: {filename}")

    file_path = None

    # Method 1: Try last_generated filepath first (most reliable)
    if last_generated['filepath']:
        candidate = Path(last_generated['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate
            print(f"   Found via last_generated: {file_path}")

    # Method 2: Search in output/bdd relative to this script
    if not file_path:
        script_dir = Path(__file__).parent.absolute()
        base_dir = script_dir / 'output' / 'bdd'

        print(f"   Searching in: {base_dir}")

        if base_dir.exists():
            for llm_dir in base_dir.iterdir():
                if llm_dir.is_dir():
                    candidate = llm_dir / filename
                    if candidate.exists():
                        file_path = candidate
                        print(f"   Found: {file_path}")
                        break

    # Method 3: Try current working directory
    if not file_path:
        base_dir = Path.cwd() / 'output' / 'bdd'
        print(f"   Searching in cwd: {base_dir}")

        if base_dir.exists():
            for llm_dir in base_dir.iterdir():
                if llm_dir.is_dir():
                    candidate = llm_dir / filename
                    if candidate.exists():
                        file_path = candidate
                        print(f"   Found: {file_path}")
                        break

    if not file_path:
        print(f"❌ File not found: {filename}")
        print(f"   last_generated: {last_generated}")
        return jsonify({'error': 'File not found'}), 404

    print(f"✅ Sending file: {file_path}")

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/llm-list')
def get_llm_list():
    """Get available LLM providers"""
    return jsonify({
        'llms': [
            {'id': 'groq', 'name': 'Groq', 'description': 'Fast & Free'},
            {'id': 'deepseek', 'name': 'DeepSeek', 'description': 'Chinese LLM'},
            {'id': 'openai', 'name': 'OpenAI', 'description': 'GPT-5 Series'},
            {'id': 'claude', 'name': 'Claude', 'description': 'Anthropic'},
            {'id': 'gemini', 'name': 'Gemini', 'description': 'Google'}
        ],
        'openai_models': [
            {'id': 'gpt-5-mini', 'name': 'GPT-5 Mini (Recommended)'},
            {'id': 'gpt-5', 'name': 'GPT-5'},
            {'id': 'gpt-5.1', 'name': 'GPT-5.1'},
            {'id': 'gpt-5.1-codex', 'name': 'GPT-5.1 Codex'}
        ]
    })


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print()
    print("=" * 60)
    print("🤖 LLM BDD Generator")
    print("=" * 60)
    print()
    print("📡 Server: http://localhost:5000")
    print()
    print("📋 Endpoints:")
    print("   GET  /              - Web interface")
    print("   POST /api/generate  - Generate BDD file")
    print("   GET  /api/download/<file>  - Download file")
    print("   GET  /api/llm-list  - Get LLM options")
    print()
    print("=" * 60)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)