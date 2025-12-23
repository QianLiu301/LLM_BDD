"""
Flask Backend for LLM Hardware Verification Demo
=================================================

Provides REST API for the HTML frontend to call Python scripts.

Run: python main.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import threading
import time

# ============================================================================
# Proxy Setup (same as feature_generator_llm.py)
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

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our generators
try:
    from feature_generator_llm import FeatureGeneratorLLM
    from metadata_extractor import MetadataExtractor
    from llm_providers import LLMFactory, LocalLLMProvider, GroqProvider
    HAS_MODULES = True

    # ========================================================================
    # HOTFIX: Force correct Groq API URL
    # The correct Groq API endpoint is /v1/, not /openai/v1/
    # ========================================================================
    if hasattr(GroqProvider, '__init__'):
        _original_groq_init = GroqProvider.__init__

        def _patched_groq_init(self, api_key=None, model="llama-3.3-70b-versatile"):
            _original_groq_init(self, api_key, model)
            # Force correct URL regardless of what's in llm_providers.py
            self.api_url = "https://api.groq.com/v1/chat/completions"
            print(f"   🔧 Groq API URL set to: {self.api_url}")

        GroqProvider.__init__ = _patched_groq_init
        print("✅ Groq URL hotfix applied (forced to /v1/ endpoint)")
    # ========================================================================

except ImportError as e:
    print(f"⚠️  Warning: Could not import modules: {e}")
    HAS_MODULES = False

app = Flask(__name__,
            static_folder='.',
            template_folder='.')
CORS(app)  # Enable CORS for API calls

# Global status for tracking pipeline progress
pipeline_status = {
    'current_step': 0,
    'total_steps': 5,
    'status': 'idle',
    'message': '',
    'results': {},
    'error': None
}


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'llm_verification_demo.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current pipeline status"""
    return jsonify(pipeline_status)


@app.route('/api/generate-feature', methods=['POST'])
def generate_feature():
    """
    Generate BDD Feature file using LLM

    Request body:
    {
        "llm": "groq",
        "model": "gpt-5-mini",  // optional, only for openai
        "input": "Generate tests for 16-bit ADD operation",
        "api_key": "xxx"  // optional
    }
    """
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported'
        }), 500

    try:
        data = request.json
        llm_name = data.get('llm', 'groq')
        user_input = data.get('input', '')
        model = data.get('model')
        api_key = data.get('api_key')

        if not user_input:
            return jsonify({
                'success': False,
                'error': 'No input provided'
            }), 400

        # Update status
        pipeline_status['current_step'] = 1
        pipeline_status['status'] = 'running'
        pipeline_status['message'] = f'Calling {llm_name.upper()} LLM...'

        # Create LLM provider
        llm_config = {}
        if api_key:
            llm_config['api_key'] = api_key
        if model:
            llm_config['model'] = model

        llm = LLMFactory.create_provider(llm_name, **llm_config)

        # Create generator
        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            debug=False
        )
        generator.llm = llm
        generator.llm_name = llm_name

        # Generate feature
        pipeline_status['message'] = 'Generating feature file...'
        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            pipeline_status['status'] = 'error'
            pipeline_status['error'] = 'Failed to generate feature'
            return jsonify({
                'success': False,
                'error': 'Failed to generate feature'
            }), 500

        # Read feature content
        with open(feature_path, 'r', encoding='utf-8') as f:
            feature_content = f.read()

        # Update results
        pipeline_status['current_step'] = 2
        pipeline_status['status'] = 'success'
        pipeline_status['message'] = 'Feature generated successfully'
        pipeline_status['results']['feature_path'] = str(feature_path)
        pipeline_status['results']['feature_preview'] = feature_content[:500]

        return jsonify({
            'success': True,
            'feature_path': str(feature_path),
            'feature_preview': feature_content[:500],
            'llm': llm_name,
            'model': model if model else llm_name
        })

    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/extract-metadata', methods=['POST'])
def extract_metadata():
    """
    Extract metadata from Feature file

    Request body:
    {
        "feature_path": "path/to/file.feature"
    }
    """
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported'
        }), 500

    try:
        # Update status
        pipeline_status['current_step'] = 3
        pipeline_status['status'] = 'running'
        pipeline_status['message'] = 'Extracting metadata...'

        # Create extractor
        extractor = MetadataExtractor(debug=False)

        # Extract from all feature files
        stats = extractor.extract_all()

        # Update results
        pipeline_status['current_step'] = 3
        pipeline_status['message'] = f'Extracted metadata from {stats["success"]} files'
        pipeline_status['results']['metadata_stats'] = stats

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-testbench', methods=['POST'])
def generate_testbench():
    """
    Generate Verilog testbench from Feature file
    """
    try:
        # Update status
        pipeline_status['current_step'] = 4
        pipeline_status['status'] = 'running'
        pipeline_status['message'] = 'Generating testbench...'

        # Run testbench generator
        result = subprocess.run(
            [sys.executable, 'testbench_generator.py'],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise Exception(f"Testbench generation failed: {result.stderr}")

        # Update results
        pipeline_status['current_step'] = 4
        pipeline_status['message'] = 'Testbench generated successfully'
        pipeline_status['results']['testbench_output'] = result.stdout

        return jsonify({
            'success': True,
            'output': result.stdout
        })

    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """
    Run simulation and generate waveform
    """
    try:
        # Update status
        pipeline_status['current_step'] = 5
        pipeline_status['status'] = 'running'
        pipeline_status['message'] = 'Running simulation...'

        # Run simulation
        result = subprocess.run(
            [sys.executable, 'simulation_runner.py'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise Exception(f"Simulation failed: {result.stderr}")

        # Update results
        pipeline_status['current_step'] = 5
        pipeline_status['status'] = 'completed'
        pipeline_status['message'] = 'Simulation completed!'
        pipeline_status['results']['simulation_output'] = result.stdout

        return jsonify({
            'success': True,
            'output': result.stdout
        })

    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    """
    Run the complete pipeline in one call

    This is more convenient for demo purposes
    """
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported'
        }), 500

    def run_async():
        global pipeline_status

        try:
            data = request.json
            llm_name = data.get('llm', 'groq')
            user_input = data.get('input', '')
            model = data.get('model')
            api_key = data.get('api_key')

            # Reset status
            pipeline_status = {
                'current_step': 0,
                'total_steps': 5,
                'status': 'running',
                'message': 'Starting pipeline...',
                'results': {},
                'error': None
            }

            # Step 1: Generate Feature
            pipeline_status['current_step'] = 1
            pipeline_status['message'] = f'Step 1/5: Calling {llm_name.upper()} LLM...'

            llm_config = {}
            if api_key:
                llm_config['api_key'] = api_key
            if model:
                llm_config['model'] = model

            llm = LLMFactory.create_provider(llm_name, **llm_config)

            generator = FeatureGeneratorLLM(
                llm_provider=llm_name,
                debug=False
            )
            generator.llm = llm
            generator.llm_name = llm_name

            feature_path = generator.generate_feature(user_input)

            if not feature_path:
                raise Exception('Failed to generate feature')

            with open(feature_path, 'r', encoding='utf-8') as f:
                feature_content = f.read()

            pipeline_status['results']['feature_path'] = str(feature_path)
            pipeline_status['results']['feature_preview'] = feature_content[:500]

            time.sleep(0.5)  # Give UI time to update

            # Step 2: Extract Metadata
            pipeline_status['current_step'] = 2
            pipeline_status['message'] = 'Step 2/5: Extracting metadata...'

            extractor = MetadataExtractor(debug=False)
            stats = extractor.extract_all()

            pipeline_status['results']['metadata_stats'] = stats

            time.sleep(0.5)

            # Step 3: Generate Testbench
            pipeline_status['current_step'] = 3
            pipeline_status['message'] = 'Step 3/5: Generating testbench...'

            result = subprocess.run(
                [sys.executable, 'testbench_generator.py'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                pipeline_status['results']['testbench_output'] = result.stdout[:500]

            time.sleep(0.5)

            # Step 4: Run Simulation
            pipeline_status['current_step'] = 4
            pipeline_status['message'] = 'Step 4/5: Running simulation...'

            result = subprocess.run(
                [sys.executable, 'simulation_runner.py'],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                pipeline_status['results']['simulation_output'] = result.stdout[:500]

            time.sleep(0.5)

            # Complete
            pipeline_status['current_step'] = 5
            pipeline_status['status'] = 'completed'
            pipeline_status['message'] = '✅ Pipeline completed successfully!'

        except Exception as e:
            pipeline_status['status'] = 'error'
            pipeline_status['error'] = str(e)
            import traceback
            traceback.print_exc()

    # Run in background thread
    thread = threading.Thread(target=run_async)
    thread.start()

    return jsonify({
        'success': True,
        'message': 'Pipeline started. Use /api/status to check progress.'
    })


@app.route('/api/execute-step', methods=['POST'])
def execute_step():
    """
    统一的步骤执行接口

    Request body:
    {
        "step": "llm|metadata|testbench|simulation|waveform",
        "llm": "groq",  // for llm step
        "model": "gpt-5-mini",  // for openai
        "input": "test requirements"  // for llm step
    }
    """
    try:
        data = request.json
        step = data.get('step')

        if step == 'llm':
            return execute_llm_step(data)
        elif step == 'metadata':
            return execute_metadata_step(data)
        elif step == 'testbench':
            return execute_testbench_step(data)
        elif step == 'simulation':
            return execute_simulation_step(data)
        elif step == 'waveform':
            return execute_waveform_step(data)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown step: {step}'
            }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def execute_llm_step(data):
    """执行LLM生成步骤"""
    llm_name = data.get('llm', 'groq')
    user_input = data.get('input', '')
    model = data.get('model')
    api_key = data.get('api_key')

    print(f"\n{'='*70}")
    print(f"🚀 [STEP 1] LLM Generation Started")
    print(f"{'='*70}")
    print(f"   LLM Provider: {llm_name.upper()}")
    print(f"   User Input: {user_input[:50]}...")
    if model:
        print(f"   Model: {model}")
    print()

    if not user_input:
        print("❌ Error: No input provided")
        return jsonify({'success': False, 'error': 'No input provided'}), 400

    try:
        # Create LLM provider
        llm_config = {}
        if api_key:
            llm_config['api_key'] = api_key
        if model:
            llm_config['model'] = model

        print(f"🔧 Creating {llm_name} provider...")
        llm = LLMFactory.create_provider(llm_name, **llm_config)
        print(f"✅ Provider created successfully")

        # Create generator
        print(f"🔧 Initializing Feature Generator...")
        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            debug=True
        )
        generator.llm = llm
        generator.llm_name = llm_name
        print(f"✅ Generator initialized")

        # Generate feature
        print(f"\n📡 Calling LLM API...")
        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            print("❌ Feature generation returned None")
            return jsonify({
                'success': False,
                'error': 'Feature generation failed - LLM returned empty response'
            }), 500

        # Verify file exists
        feature_path_obj = Path(feature_path)
        if not feature_path_obj.exists():
            print(f"❌ File does not exist: {feature_path}")
            return jsonify({
                'success': False,
                'error': f'Feature file was not created: {feature_path}'
            }), 500

        # Verify file has content
        file_size = feature_path_obj.stat().st_size
        if file_size == 0:
            print(f"❌ File is empty: {feature_path}")
            return jsonify({
                'success': False,
                'error': 'Generated feature file is empty'
            }), 500

        print(f"✅ File verified: {feature_path} ({file_size} bytes)")

        # Read preview
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\n{'='*70}")
        print(f"✅ [STEP 1] LLM Generation Completed Successfully")
        print(f"{'='*70}")
        print(f"   File: {feature_path_obj.name}")
        print(f"   Size: {file_size} bytes")
        print(f"   Location: {feature_path}")
        print(f"{'='*70}\n")

        return jsonify({
            'success': True,
            'message': f'Feature generated by {llm_name.upper()}',
            'files': [feature_path_obj.name],
            'file_path': str(feature_path),
            'output': content[:500] + ('...' if len(content) > 500 else '')
        })

    except Exception as e:
        print(f"\n❌ Exception in execute_llm_step:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }), 500


def execute_metadata_step(data):
    """执行Metadata提取步骤"""
    extractor = MetadataExtractor(debug=False)
    stats = extractor.extract_all()

    # Get generated files
    metadata_files = []
    metadata_dir = Path('output/metadata')
    if metadata_dir.exists():
        for llm_dir in metadata_dir.iterdir():
            if llm_dir.is_dir():
                metadata_files.extend([f.name for f in llm_dir.glob('*.json')])
                metadata_files.extend([f.name for f in llm_dir.glob('*.txt')])

    return jsonify({
        'success': True,
        'message': f'Metadata extracted from {stats["success"]} files',
        'files': metadata_files[:5],  # First 5 files
        'output': f'Total: {stats["total"]}, Success: {stats["success"]}, Failed: {stats["failed"]}\nLLM Providers: {", ".join(stats["llm_providers"])}'
    })


def execute_testbench_step(data):
    """执行Testbench生成步骤"""
    # Run testbench_generator.py
    result = subprocess.run(
        [sys.executable, 'testbench_generator.py'],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        return jsonify({
            'success': False,
            'error': f'Testbench generation failed:\n{result.stderr}'
        }), 500

    # Get generated testbench files
    testbench_files = []
    verilog_dir = Path('output/verilog')
    if verilog_dir.exists():
        for llm_dir in verilog_dir.iterdir():
            if llm_dir.is_dir():
                testbench_files.extend([f.name for f in llm_dir.glob('*_tb.v')])

    return jsonify({
        'success': True,
        'message': 'Testbench generated successfully',
        'files': testbench_files[:5],
        'output': result.stdout[:500]
    })


def execute_simulation_step(data):
    """执行仿真步骤"""
    # Check if simulation_controller.py exists
    if not Path('simulation_controller.py').exists():
        return jsonify({
            'success': False,
            'error': 'simulation_controller.py not found'
        }), 404

    # Run simulation
    result = subprocess.run(
        [sys.executable, 'simulation_controller.py'],
        capture_output=True,
        text=True,
        timeout=180
    )

    if result.returncode != 0:
        return jsonify({
            'success': False,
            'error': f'Simulation failed:\n{result.stderr}'
        }), 500

    # Get generated VCD files
    vcd_files = []
    simulation_dir = Path('output/simulation')
    if simulation_dir.exists():
        for vcd_file in simulation_dir.rglob('*.vcd'):
            vcd_files.append(vcd_file.name)

    return jsonify({
        'success': True,
        'message': 'Simulation completed successfully',
        'files': vcd_files,
        'output': result.stdout[:500]
    })


def execute_waveform_step(data):
    """执行波形查看步骤"""
    # Check if gtkwave_controller.py exists
    if not Path('gtkwave_controller.py').exists():
        return jsonify({
            'success': False,
            'error': 'gtkwave_controller.py not found'
        }), 404

    # Run GTKWave viewer
    result = subprocess.run(
        [sys.executable, 'gtkwave_controller.py'],
        capture_output=True,
        text=True,
        timeout=10
    )

    return jsonify({
        'success': True,
        'message': 'GTKWave viewer launched',
        'output': 'Waveform viewer opened in separate window'
    })


@app.route('/api/download/<step>/<path:filename>')
def download_file(step, filename):
    """
    下载生成的文件

    Args:
        step: llm|metadata|testbench|simulation
        filename: 文件名
    """
    print(f"\n{'='*70}")
    print(f"📥 Download Request")
    print(f"{'='*70}")
    print(f"   Step: {step}")
    print(f"   Filename: {filename}")

    # Map step to directory
    dir_map = {
        'llm': 'output/bdd',
        'metadata': 'output/metadata',
        'testbench': 'output/verilog',
        'simulation': 'output/simulation'
    }

    if step not in dir_map:
        print(f"❌ Invalid step: {step}")
        return jsonify({'error': 'Invalid step'}), 400

    base_dir = Path(dir_map[step])
    print(f"   Base directory: {base_dir.absolute()}")

    # Check if base directory exists
    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir.absolute()}")
        return jsonify({'error': f'Directory not found: {base_dir}'}), 404

    # Search for file in subdirectories
    print(f"   Searching for file...")
    file_path = None
    all_files = []

    for subdir in base_dir.rglob(filename):
        all_files.append(str(subdir))
        if subdir.is_file():
            file_path = subdir
            break

    if not file_path:
        print(f"❌ File not found: {filename}")
        print(f"   Searched in: {base_dir.absolute()}")
        print(f"   Found files matching pattern:")
        for f in all_files[:5]:
            print(f"     - {f}")
        return jsonify({
            'error': 'File not found',
            'searched_in': str(base_dir.absolute()),
            'filename': filename
        }), 404

    print(f"✅ File found: {file_path}")
    print(f"{'='*70}\n")

    # Send file
    return send_from_directory(
        file_path.parent,
        file_path.name,
        as_attachment=True
    )


@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    """Reset pipeline status"""
    global pipeline_status
    pipeline_status = {
        'current_step': 0,
        'total_steps': 5,
        'status': 'idle',
        'message': '',
        'results': {},
        'error': None
    }
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 LLM Hardware Verification Demo Server")
    print("=" * 70)
    print()
    print("📡 Server starting on http://localhost:5000")
    print()
    print("📋 Available endpoints:")
    print("   GET  /              - Main HTML page")
    print("   GET  /api/status    - Get pipeline status")
    print("   POST /api/generate-feature    - Generate feature file")
    print("   POST /api/extract-metadata    - Extract metadata")
    print("   POST /api/generate-testbench  - Generate testbench")
    print("   POST /api/run-simulation      - Run simulation")
    print("   POST /api/run-pipeline        - Run complete pipeline")
    print("   POST /api/reset               - Reset status")
    print()
    print("💡 Open http://localhost:5000 in your browser")
    print("=" * 70)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)