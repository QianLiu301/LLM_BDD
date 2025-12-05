#!/usr/bin/env python3
"""
Simulation Controller - Run Verilog simulations with iverilog/vvp
=================================================================

This module scans for ALU design files and their corresponding testbenches,
compiles and runs simulations, and optionally opens waveform viewer.

ARCHITECTURE:
    output/verilog/
    ├── alu_8bit.v       ──┐
    ├── alu_8bit_tb.v    ──┼──► iverilog ──► vvp ──► output/simulation/
    ├── alu_16bit.v      ──┤
    ├── alu_16bit_tb.v   ──┤
    └── ...              ──┘

KEY FEATURES:
- Automatically pairs DUT with its corresponding testbench
- Supports multiple bitwidth ALUs (8, 16, 32, 64 bit)
- Generates VCD waveform files for GTKWave
- Interactive VCD file selection
- Cross-platform path handling

NO LLM REQUIRED - This is a deterministic simulation runner.
"""

import subprocess
import sys
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class SimulationController:
    """
    Controller for running Verilog simulations.

    Scans for DUT and testbench pairs, compiles with iverilog,
    runs with vvp, and optionally opens GTKWave for waveform viewing.
    """

    def __init__(
            self,
            verilog_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            project_root: Optional[str] = None,
            debug: bool = True
    ):
        """
        Initialize the simulation controller.

        Args:
            verilog_dir: Directory containing .v and *_tb.v files
            output_dir: Directory for simulation output (.vvp, .vcd files)
            project_root: Project root directory for path resolution
            debug: Enable debug output
        """
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None

        # Setup paths
        self.verilog_dir = self._find_verilog_dir(verilog_dir)
        self.output_dir = self._setup_output_dir(output_dir)

        print("=" * 70)
        print("Simulation Controller")
        print("=" * 70)
        print(f"📁 Verilog directory: {self.verilog_dir}")
        print(f"📁 Output directory:  {self.output_dir}")

    def _find_verilog_dir(self, verilog_dir: Optional[str]) -> Path:
        """Find Verilog source directory"""
        # 1. Explicit specification
        if verilog_dir:
            path = Path(verilog_dir)
            if path.exists():
                return path
            print(f"⚠️  Specified verilog_dir not found: {verilog_dir}")

        # 2. Project root based
        if self.project_root:
            path = self.project_root / "output" / "verilog"
            if path.exists():
                return path

        # 3. Search common locations
        current = Path.cwd()
        search_paths = [
            current / "output" / "verilog",
            current / "verilog",
            current.parent / "output" / "verilog",
            current / "src" / "output" / "verilog",
        ]

        for path in search_paths:
            if path.exists():
                self._debug_print(f"Found verilog at: {path}", "SUCCESS")
                return path

        # 4. Fallback to absolute path (Windows)
        fallback_path = Path(r"D:\DE\RQ\MultiLLM_BDD_Comparison\HdlFormalVerifier\output\verilog")
        if fallback_path.exists():
            return fallback_path

        # 5. Create default
        default = current / "output" / "verilog"
        print(f"⚠️  No verilog directory found. Please specify with --verilog-dir")
        return default

    def _setup_output_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory for simulation results"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "simulation"
        else:
            # Same level as verilog directory
            path = self.verilog_dir.parent / "simulation"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _debug_print(self, message: str, level: str = "INFO"):
        """Debug output"""
        if not self.debug and level == "DEBUG":
            return

        icons = {
            "INFO": "ℹ️ ", "DEBUG": "🔍", "WARN": "⚠️ ",
            "ERROR": "❌", "SUCCESS": "✅", "STEP": "📌",
        }
        icon = icons.get(level, "  ")
        print(f"   {icon} [{level}] {message}")

    def check_tools(self) -> Dict[str, bool]:
        """Check whether required tools are installed"""
        print("\n🔧 Checking simulation tools...")

        tools = {
            "iverilog": False,
            "vvp": False,
            "gtkwave": False,
        }

        # Check iverilog
        try:
            result = subprocess.run(
                ["iverilog", "-V"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                tools["iverilog"] = True
                version = result.stdout.split("\n")[0]
                tools["iverilog_version"] = version
                print(f"   ✅ iverilog: {version}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   ❌ iverilog: not found")

        # Check vvp
        try:
            result = subprocess.run(
                ["vvp", "-V"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                tools["vvp"] = True
                print("   ✅ vvp: installed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   ❌ vvp: not found")

        # Check gtkwave
        try:
            result = subprocess.run(
                ["gtkwave", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            tools["gtkwave"] = True
            print("   ✅ gtkwave: installed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   ⚠️  gtkwave: not found (waveform viewing disabled)")

        return tools

    def find_dut_testbench_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching DUT and testbench file pairs.

        Matching logic:
            alu_8bit.v  ↔ alu_8bit_tb.v
            alu_16bit.v ↔ alu_16bit_tb.v
            etc.

        Returns:
            List of (dut_file, testbench_file) tuples
        """
        print(f"\n🔍 Scanning for Verilog files in: {self.verilog_dir}")

        if not self.verilog_dir.exists():
            print(f"   ❌ Directory not found: {self.verilog_dir}")
            return []

        # Find all .v files
        all_v_files = list(self.verilog_dir.glob("*.v"))

        # Classify files
        testbench_files: Dict[str, Path] = {}  # base_name -> tb_file
        design_files: Dict[str, Path] = {}  # base_name -> dut_file

        for v_file in all_v_files:
            name = v_file.stem  # filename without extension

            # Check if it's a testbench file
            if name.endswith("_tb"):
                # Extract base name: alu_8bit_tb -> alu_8bit
                base_name = name[:-3]
                testbench_files[base_name] = v_file
            elif "_tb" not in name and "test" not in name.lower():
                # It's a design file
                design_files[name] = v_file

        # Match pairs
        pairs: List[Tuple[Path, Path]] = []

        print(f"\n📋 Found files:")
        print(f"   Design files:    {len(design_files)}")
        print(f"   Testbench files: {len(testbench_files)}")

        print(f"\n🔗 Matching DUT ↔ Testbench pairs:")

        for base_name, dut_file in sorted(design_files.items()):
            if base_name in testbench_files:
                tb_file = testbench_files[base_name]
                pairs.append((dut_file, tb_file))
                print(f"   ✅ {dut_file.name} ↔ {tb_file.name}")
            else:
                print(f"   ⚠️  {dut_file.name} (no matching testbench)")

        # Check for orphan testbenches
        for base_name, tb_file in testbench_files.items():
            if base_name not in design_files:
                print(f"   ⚠️  {tb_file.name} (no matching DUT)")

        return pairs

    def compile_verilog(self, dut_file: Path, tb_file: Path) -> Optional[Path]:
        """
        Compile a DUT + testbench pair.

        Args:
            dut_file: Path to DUT file (e.g., alu_8bit.v)
            tb_file: Path to testbench file (e.g., alu_8bit_tb.v)

        Returns:
            Path to compiled .vvp file, or None if compilation failed
        """
        # Output filename based on testbench
        output_vvp = self.output_dir / f"{tb_file.stem}.vvp"

        # Build compile command
        compile_cmd = [
            "iverilog",
            "-g2012",  # SystemVerilog 2012 support
            "-o", str(output_vvp),
            str(dut_file),  # DUT first
            str(tb_file),  # Testbench second
        ]

        print(f"\n📦 Compiling: {dut_file.name} + {tb_file.name}")
        self._debug_print(f"Command: {' '.join(compile_cmd)}", "DEBUG")

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print(f"   ✅ Compiled: {output_vvp.name}")
                if result.stderr:
                    # iverilog warnings go to stderr
                    warnings = result.stderr.strip()
                    if warnings:
                        print(f"   ⚠️  Warnings:\n{warnings}")
                return output_vvp
            else:
                print(f"   ❌ Compilation failed")
                print(f"   Error:\n{result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("   ❌ Compilation timeout")
            return None
        except FileNotFoundError:
            print("   ❌ iverilog not found. Please install Icarus Verilog.")
            return None
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return None

    def run_simulation(self, vvp_file: Path) -> Tuple[bool, Optional[Path]]:
        """
        Run simulation using vvp.

        Args:
            vvp_file: Path to compiled .vvp file

        Returns:
            (success, vcd_file_path)
        """
        if not vvp_file.exists():
            print(f"   ❌ VVP file not found: {vvp_file}")
            return False, None

        print(f"\n🚀 Running simulation: {vvp_file.name}")

        try:
            result = subprocess.run(
                ["vvp", str(vvp_file)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.output_dir),  # Run in output dir so VCD goes there
            )

            if result.returncode == 0:
                print("   ✅ Simulation completed")
                print("\n" + "─" * 60)
                print("SIMULATION OUTPUT:")
                print("─" * 60)
                print(result.stdout)
                print("─" * 60)

                # Find generated VCD file
                vcd_name = vvp_file.stem.replace("_tb", "") + "_tb.vcd"
                vcd_file = self.output_dir / vcd_name

                # Also check for alternative VCD names
                if not vcd_file.exists():
                    vcd_files = list(self.output_dir.glob("*.vcd"))
                    # Get most recently modified
                    if vcd_files:
                        vcd_file = max(vcd_files, key=lambda f: f.stat().st_mtime)

                if vcd_file.exists():
                    print(f"\n   📊 VCD waveform: {vcd_file.name}")
                    return True, vcd_file
                else:
                    print(f"\n   ⚠️  No VCD file generated")
                    return True, None
            else:
                print(f"   ❌ Simulation failed")
                print(f"   Error:\n{result.stderr}")
                return False, None

        except subprocess.TimeoutExpired:
            print("   ❌ Simulation timeout (>60s)")
            return False, None
        except FileNotFoundError:
            print("   ❌ vvp not found")
            return False, None
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False, None

    def open_gtkwave(self, vcd_file: Path) -> bool:
        """Open a VCD file in GTKWave"""
        try:
            print(f"\n🌊 Opening waveform: {vcd_file.name}")
            subprocess.Popen(["gtkwave", str(vcd_file)])
            return True
        except FileNotFoundError:
            print("   ❌ GTKWave not found")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def interactive_vcd_selection(self, vcd_files: List[Path]) -> Optional[Path]:
        """
        Interactive VCD file selection menu.

        Args:
            vcd_files: List of available VCD files

        Returns:
            Selected VCD file path, or None if cancelled
        """
        if not vcd_files:
            print("\n⚠️  No VCD files available")
            return None

        print("\n" + "=" * 60)
        print("📊 Available VCD Waveform Files:")
        print("=" * 60)

        for i, vcd in enumerate(vcd_files, 1):
            size_kb = vcd.stat().st_size / 1024
            print(f"  {i}. {vcd.name} ({size_kb:.1f} KB)")

        print(f"  A. Open ALL files")
        print(f"  0. Skip / Cancel")
        print("=" * 60)

        try:
            choice = input("\nSelect file to open [1]: ").strip().upper()

            if not choice:
                choice = "1"

            if choice == "0":
                print("   Skipped.")
                return None

            if choice == "A":
                for vcd in vcd_files:
                    self.open_gtkwave(vcd)
                return vcd_files[0] if vcd_files else None

            try:
                index = int(choice) - 1
                if 0 <= index < len(vcd_files):
                    selected = vcd_files[index]
                    self.open_gtkwave(selected)
                    return selected
                else:
                    print(f"   Invalid selection: {choice}")
                    return None
            except ValueError:
                print(f"   Invalid input: {choice}")
                return None

        except KeyboardInterrupt:
            print("\n   Cancelled.")
            return None

    def run_all_simulations(self, auto_wave: str = "prompt") -> bool:
        """
        Run complete simulation flow for all DUT/testbench pairs.

        Args:
            auto_wave: How to handle waveform viewing
                - "no": Don't open GTKWave
                - "prompt": Ask user which to open
                - "all": Open all VCD files
                - "first": Open first VCD file only

        Returns:
            True if at least one simulation succeeded
        """
        print("\n" + "=" * 70)
        print("🔬 SIMULATION FLOW")
        print("=" * 70)

        # 1. Check tools
        tools = self.check_tools()

        if not tools["iverilog"] or not tools["vvp"]:
            print("\n❌ Required tools not installed. Please install Icarus Verilog.")
            print("   Windows: Download from http://bleyer.org/icarus/")
            print("   Linux:   sudo apt install iverilog")
            print("   macOS:   brew install icarus-verilog")
            return False

        # 2. Find DUT + testbench pairs
        pairs = self.find_dut_testbench_pairs()

        if not pairs:
            print("\n❌ No matching DUT/testbench pairs found")
            return False

        # 3. Compile and simulate each pair
        results = []
        vcd_files = []

        for i, (dut_file, tb_file) in enumerate(pairs, 1):
            print(f"\n{'─' * 70}")
            print(f"📋 Test {i}/{len(pairs)}: {dut_file.stem}")
            print(f"{'─' * 70}")

            # Compile
            vvp_file = self.compile_verilog(dut_file, tb_file)
            if vvp_file is None:
                results.append((dut_file.stem, False, "Compilation failed"))
                continue

            # Simulate
            success, vcd_file = self.run_simulation(vvp_file)

            if success:
                results.append((dut_file.stem, True, "PASS"))
                if vcd_file:
                    vcd_files.append(vcd_file)
            else:
                results.append((dut_file.stem, False, "Simulation failed"))

        # 4. Summary
        print("\n" + "=" * 70)
        print("📊 SIMULATION SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, success, _ in results if success)
        failed = len(results) - passed

        print(f"\n   Total:  {len(results)}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} {'❌' if failed > 0 else ''}")

        print(f"\n   Results:")
        for name, success, msg in results:
            icon = "✅" if success else "❌"
            print(f"      {icon} {name}: {msg}")

        # 5. Output files
        print(f"\n📁 Output directory: {self.output_dir}")

        vvp_files = list(self.output_dir.glob("*.vvp"))
        all_vcd_files = list(self.output_dir.glob("*.vcd"))

        if vvp_files:
            print(f"\n   Compiled simulations (.vvp):")
            for f in vvp_files:
                print(f"      • {f.name}")

        if all_vcd_files:
            print(f"\n   Waveform files (.vcd):")
            for f in all_vcd_files:
                size_kb = f.stat().st_size / 1024
                print(f"      • {f.name} ({size_kb:.1f} KB)")

        # 6. Waveform viewing
        if passed > 0 and tools.get("gtkwave") and all_vcd_files:
            if auto_wave == "all":
                print("\n🌊 Opening all waveforms...")
                for vcd in all_vcd_files:
                    self.open_gtkwave(vcd)
            elif auto_wave == "first":
                self.open_gtkwave(all_vcd_files[0])
            elif auto_wave == "prompt":
                self.interactive_vcd_selection(all_vcd_files)
            # else: "no" - don't open

        print("\n" + "=" * 70)
        if passed == len(results):
            print("✨ All simulations completed successfully!")
        elif passed > 0:
            print(f"⚠️  {passed}/{len(results)} simulations passed")
        else:
            print("❌ All simulations failed")
        print("=" * 70)

        return passed > 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Verilog Simulation Controller - Compile and run ALU testbenches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default paths
  python simulation_controller.py

  # Specify directories
  python simulation_controller.py --verilog-dir ./output/verilog --output-dir ./output/simulation

  # Auto-open all waveforms
  python simulation_controller.py --open-wave all

  # Don't open waveforms
  python simulation_controller.py --open-wave no

  # With project root
  python simulation_controller.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog

Directory Structure:
  output/
  ├── verilog/              ← Input: DUT + Testbench files
  │   ├── alu_8bit.v
  │   ├── alu_8bit_tb.v
  │   ├── alu_16bit.v
  │   └── alu_16bit_tb.v
  └── simulation/           ← Output: Compiled + Waveform files
      ├── alu_8bit_tb.vvp
      ├── alu_8bit_tb.vcd
      ├── alu_16bit_tb.vvp
      └── alu_16bit_tb.vcd

Required Tools:
  - iverilog (Icarus Verilog compiler)
  - vvp (Verilog simulation runtime)
  - gtkwave (optional, for waveform viewing)
        '''
    )

    # Default path for your project
    default_verilog = r"D:\DE\RQ\MultiLLM_BDD_Comparison\HdlFormalVerifier\output\verilog"
    default_output = r"D:\DE\RQ\MultiLLM_BDD_Comparison\HdlFormalVerifier\output\simulation"

    parser.add_argument(
        '--verilog-dir', '-v',
        help='Directory containing Verilog files (DUT + testbench)',
        default=default_verilog
    )

    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for simulation results',
        default=default_output
    )

    parser.add_argument(
        '--project-root', '-p',
        help='Project root directory',
        default=None
    )

    parser.add_argument(
        '--open-wave', '-w',
        choices=['no', 'prompt', 'all', 'first'],
        default='prompt',
        help='''Waveform viewing mode:
            no: Don't open GTKWave
            prompt: Ask which file to open (default)
            all: Open all VCD files
            first: Open first VCD file only'''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Enable debug output'
    )

    args = parser.parse_args()

    try:
        # Create controller
        controller = SimulationController(
            verilog_dir=args.verilog_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            debug=args.debug
        )

        # Run all simulations
        success = controller.run_all_simulations(auto_wave=args.open_wave)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()