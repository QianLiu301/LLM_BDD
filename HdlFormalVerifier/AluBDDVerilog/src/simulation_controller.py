#!/usr/bin/env python3
"""
Dynamic Simulation Controller - Using existing Verilog files

Supports reading .v and *_tb.v files from a given directory and running simulations.
Version 2.0 - Enhanced with interactive VCD selection
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


class SimulationController:
    def __init__(self, input_dir: str, output_dir: str = None):
        """
        Initialize the simulation controller.

        Args:
            input_dir: Input directory containing .v and *_tb.v files
            output_dir: Output directory (optional). If None, a 'simulation_output'
                        folder will be created inside the input directory.
        """
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        # Set output directory
        if output_dir is None:
            self.output_dir = self.input_dir / "simulation_output"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")

    def find_verilog_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Scan the input directory to find Verilog files.

        Returns:
            (design_files, testbench_files): lists of design files and testbench files
        """
        print(f"\nScanning directory: {self.input_dir}")

        # Find all .v files
        all_v_files = list(self.input_dir.glob("*.v"))

        # Classify files
        testbench_files: List[Path] = []
        design_files: List[Path] = []

        for v_file in all_v_files:
            # Identify testbench files (usually contain tb, test, testbench, _tb)
            name_lower = v_file.stem.lower()
            if any(keyword in name_lower for keyword in ["tb", "test", "testbench", "_tb"]):
                testbench_files.append(v_file)
            else:
                design_files.append(v_file)

        print(f"\nFound files:")
        print(f"  Design files ({len(design_files)}):")
        for f in design_files:
            print(f"    - {f.name}")
        print(f"  Testbench files ({len(testbench_files)}):")
        for f in testbench_files:
            print(f"    - {f.name}")

        return design_files, testbench_files

    def check_tools(self) -> dict:
        """Check whether required tools are installed."""
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
                print(f"✓ iverilog: {version}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("✗ iverilog: not found")

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
                print("✓ vvp: installed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("✗ vvp: not found")

        # Check gtkwave
        try:
            result = subprocess.run(
                ["gtkwave", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            tools["gtkwave"] = True
            print("✓ gtkwave: installed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("✗ gtkwave: not found")

        return tools

    def compile_verilog(self, design_files: List[Path], testbench_file: Path) -> Optional[Path]:
        """
        Compile Verilog files.

        Args:
            design_files: List of design files
            testbench_file: Testbench file

        Returns:
            Path to the compiled .vvp output file, or None if compilation failed.
        """
        # Generate output filename based on testbench filename
        output_sim = self.output_dir / f"{testbench_file.stem}.vvp"

        # Build compile command - use -g2012 for SystemVerilog support
        compile_cmd = ["iverilog", "-g2012", "-o", str(output_sim)]

        # Add design files
        for design_file in design_files:
            compile_cmd.append(str(design_file))

        # Add testbench file
        compile_cmd.append(str(testbench_file))

        print(f"\nCompiling Verilog files...")
        print(f"Compile command: {' '.join(compile_cmd)}")

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print(f"✓ Compilation succeeded: {output_sim.name}")
                if result.stdout:
                    print(f"  Compiler output: {result.stdout.strip()}")
                return output_sim
            else:
                print("✗ Compilation failed")
                print(f"  Error message:\n{result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("✗ Compilation timeout")
            return None
        except FileNotFoundError:
            print("✗ iverilog command not found. Please ensure Icarus Verilog is installed.")
            return None
        except Exception as e:
            print(f"✗ Exception during compilation: {e}")
            return None

    def run_simulation(self, vvp_file: Path) -> bool:
        """
        Run the simulation.

        Args:
            vvp_file: Path to the compiled .vvp file

        Returns:
            True if simulation succeeded, False otherwise.
        """
        if not vvp_file.exists():
            print(f"Error: simulation file does not exist: {vvp_file}")
            return False

        print(f"\nRunning simulation: {vvp_file.name}")

        try:
            run_cmd = ["vvp", str(vvp_file)]

            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.output_dir),
            )

            if result.returncode == 0:
                print("✓ Simulation succeeded")
                print("\n--- Simulation output ---")
                print(result.stdout)
                if result.stderr:
                    print("\n--- Warnings / stderr ---")
                    print(result.stderr)
                return True
            else:
                print("✗ Simulation failed")
                print(f"Error message:\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("✗ Simulation timeout")
            return False
        except FileNotFoundError:
            print("✗ vvp command not found")
            return False
        except Exception as e:
            print(f"✗ Exception during simulation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_vcd_files(self) -> List[Path]:
        """Find .vcd files in the output directory."""
        vcd_files = list(self.output_dir.glob("*.vcd"))
        return vcd_files

    def open_gtkwave(self, vcd_file: Path = None, open_all: bool = False) -> bool:
        """
        Open GTKWave to view waveforms.

        Args:
            vcd_file: Path to the VCD file (optional)
            open_all: If True, open all VCD files

        Returns:
            True if at least one GTKWave instance was opened successfully
        """
        vcd_files = self.find_vcd_files()

        if not vcd_files:
            print(f"Warning: no .vcd files found in {self.output_dir}")
            return False

        # If open_all is True, open all VCD files
        if open_all:
            print(f"\nOpening all {len(vcd_files)} VCD files in GTKWave...")
            success_count = 0
            for vcd in vcd_files:
                print(f"  Opening: {vcd.name}")
                try:
                    subprocess.Popen(["gtkwave", str(vcd)])
                    success_count += 1
                except Exception as e:
                    print(f"    Error opening {vcd.name}: {e}")

            if success_count > 0:
                print(f"\n✓ Successfully opened {success_count}/{len(vcd_files)} VCD files")
                return True
            else:
                print("\n✗ Failed to open any VCD files")
                return False

        # If specific file is provided
        if vcd_file is not None:
            if not vcd_file.exists():
                print(f"Warning: VCD file does not exist: {vcd_file}")
                return False

            print(f"\nOpening GTKWave: {vcd_file.name}")
            try:
                subprocess.Popen(["gtkwave", str(vcd_file)])
                return True
            except FileNotFoundError:
                print("Error: GTKWave not found. Please ensure it is installed.")
                return False
            except Exception as e:
                print(f"Exception while starting GTKWave: {e}")
                return False

        # Interactive selection
        return self._interactive_vcd_selection(vcd_files)

    def _interactive_vcd_selection(self, vcd_files: List[Path]) -> bool:
        """
        Allow user to interactively select which VCD files to open.

        Args:
            vcd_files: List of available VCD files

        Returns:
            True if at least one file was opened successfully
        """
        print("\n" + "=" * 60)
        print("Available VCD files:")
        print("=" * 60)

        for i, vcd in enumerate(vcd_files, 1):
            print(f"  {i}. {vcd.name}")

        print(f"  A. Open ALL files")
        print(f"  0. Cancel")
        print("=" * 60)

        try:
            choice = input("\nSelect file(s) to open (number/A/0) [default: 1]: ").strip().upper()

            if not choice:
                choice = "1"

            # Cancel
            if choice == "0":
                print("Cancelled.")
                return False

            # Open all
            if choice == "A":
                return self.open_gtkwave(open_all=True)

            # Open specific file
            try:
                index = int(choice) - 1
                if 0 <= index < len(vcd_files):
                    selected_vcd = vcd_files[index]
                    print(f"\nOpening: {selected_vcd.name}")
                    try:
                        subprocess.Popen(["gtkwave", str(selected_vcd)])
                        return True
                    except Exception as e:
                        print(f"Error opening GTKWave: {e}")
                        return False
                else:
                    print(f"Invalid selection: {choice}")
                    return False
            except ValueError:
                print(f"Invalid input: {choice}")
                return False

        except KeyboardInterrupt:
            print("\nCancelled by user.")
            return False
        except Exception as e:
            print(f"Error during selection: {e}")
            return False

    def run_full_simulation(self, auto_open_wave: str = "prompt") -> bool:
        """
        Run the complete simulation flow.

        Args:
            auto_open_wave: How to handle opening waveforms
                - "no": Don't open GTKWave
                - "prompt": Prompt user to select (default)
                - "all": Automatically open all VCD files
                - "first": Automatically open the first VCD file

        Returns:
            True if at least one testbench simulation succeeded
        """
        print("\n" + "=" * 60)
        print("Starting simulation flow")
        print("=" * 60)

        # 1. Check tools
        print("\n1. Checking tools")
        tools = self.check_tools()

        if not tools["iverilog"]:
            print("\nError: iverilog is not installed, cannot continue.")
            return False

        # 2. Find Verilog files
        print("\n2. Searching for Verilog files")
        design_files, testbench_files = self.find_verilog_files()

        if not design_files:
            print("\nError: no design files found")
            return False

        if not testbench_files:
            print("\nError: no testbench files found")
            return False

        # 3. Compile and run each testbench
        success_count = 0
        for tb_file in testbench_files:
            print(f"\n3. Compiling testbench: {tb_file.name}")
            vvp_file = self.compile_verilog(design_files, tb_file)

            if vvp_file is None:
                print(f"  Skipping simulation for {tb_file.name} (compilation failed)")
                continue

            print(f"\n4. Running simulation: {tb_file.name}")
            if self.run_simulation(vvp_file):
                success_count += 1

        # 4. Summary
        print("\n" + "=" * 60)
        print(f"Simulation finished: {success_count}/{len(testbench_files)} tests succeeded")
        print("=" * 60)

        # Show generated files
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        vvp_files = list(self.output_dir.glob("*.vvp"))
        vcd_files = self.find_vcd_files()

        if vvp_files:
            print("  VVP files:")
            for f in vvp_files:
                print(f"    - {f.name}")

        if vcd_files:
            print("  VCD files:")
            for f in vcd_files:
                print(f"    - {f.name}")

        # Auto-open waveform viewer
        if success_count > 0 and tools.get("gtkwave"):
            # Normalize auto_open_wave parameter
            if auto_open_wave is True:
                auto_open_wave = "all"
            elif auto_open_wave is False:
                auto_open_wave = "no"
            elif auto_open_wave is None:
                auto_open_wave = "prompt"

            auto_open_wave = str(auto_open_wave).lower()

            if auto_open_wave == "all":
                print("\nAuto-opening all VCD files...")
                self.open_gtkwave(open_all=True)
            elif auto_open_wave == "first":
                print("\nAuto-opening first VCD file...")
                first_vcd = vcd_files[0] if vcd_files else None
                if first_vcd:
                    self.open_gtkwave(vcd_file=first_vcd)
            elif auto_open_wave == "prompt":
                # Interactive selection
                self.open_gtkwave()
            # else: "no" - don't open anything

        return success_count > 0


def main():
    """Main entry point."""
    print("Dynamic Simulation Controller - using existing Verilog files")
    print("=" * 60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Dynamic Verilog Simulation Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (prompt for VCD selection)
  python simulation_controller.py

  # Auto-open all VCD files
  python simulation_controller.py --open-wave all

  # Don't open any waveforms
  python simulation_controller.py --open-wave no

  # Specify input directory
  python simulation_controller.py -i ./verilog -o ./sim_output
        '''
    )

    parser.add_argument(
        '-i', '--input-dir',
        help='Input directory containing Verilog files',
        default=r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\AluBDDVerilog\src\output\verilog"
    )

    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for simulation results (default: input_dir/simulation_output)',
        default=None
    )

    parser.add_argument(
        '--open-wave',
        choices=['no', 'prompt', 'all', 'first'],
        default='prompt',
        help='''How to handle opening waveforms:
             no: Don't open GTKWave
             prompt: Prompt user to select (default)
             all: Open all VCD files
             first: Open first VCD file'''
    )

    args = parser.parse_args()

    try:
        # Create controller
        controller = SimulationController(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        # Run simulation
        success = controller.run_full_simulation(auto_open_wave=args.open_wave)

        if success:
            print("\n✓ Simulation flow completed successfully!")
        else:
            print("\n✗ Simulation flow failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()