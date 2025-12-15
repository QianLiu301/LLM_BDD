#!/usr/bin/env python3
"""
Simulation Runner - Multi-Bitwidth & Multi-LLM Support
======================================================

ENHANCED VERSION: Automatically matches testbench bitwidth with corresponding DUT

Features:
- ✅ Auto-detects bitwidth from testbench filename
- ✅ Auto-matches to correct DUT (alu_8bit.v, alu_16bit.v, etc.)
- ✅ Ignores metadata .json files
- ✅ Supports multiple bitwidths in single run
- ✅ Multi-LLM comparison

Example:
    testbench: alu_16bit_20251209_132949_tb.v → DUT: alu_16bit.v
    testbench: alu_32bit_20251210_143050_tb.v → DUT: alu_32bit.v
"""

import argparse
import subprocess
import sys
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class SimulationResult:
    """Container for simulation results"""

    def __init__(self, llm_name: str, testbench_name: str, bitwidth: int):
        self.llm_name = llm_name
        self.testbench_name = testbench_name
        self.bitwidth = bitwidth
        self.dut_file = None
        self.success = False
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.simulation_log = ""
        self.vcd_file = None
        self.compile_time = 0.0
        self.sim_time = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "llm_name": self.llm_name,
            "testbench_name": self.testbench_name,
            "bitwidth": self.bitwidth,
            "dut_file": str(self.dut_file) if self.dut_file else None,
            "success": self.success,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "compile_time": self.compile_time,
            "sim_time": self.sim_time,
        }


class SimulationRunner:
    """
    Multi-Bitwidth & Multi-LLM simulation runner.

    Automatically matches testbench bitwidth with corresponding DUT.
    """

    def __init__(
            self,
            verilog_base_dir: Optional[str] = None,
            dut_dir: Optional[str] = None,
            results_dir: Optional[str] = None,
            project_root: Optional[str] = None,
            debug: bool = True
    ):
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None

        # Setup paths
        self.verilog_base_dir = self._find_verilog_base_dir(verilog_base_dir)
        self.dut_dir = self._find_dut_dir(dut_dir)
        self.results_base_dir = self._setup_results_dir(results_dir)

        # Cache available DUTs
        self.available_duts = self._scan_available_duts()

        print("=" * 70)
        print("🔬 Multi-Bitwidth & Multi-LLM Simulation Runner")
        print("=" * 70)
        print(f"📁 Verilog base directory: {self.verilog_base_dir}")
        print(f"📁 DUT directory:          {self.dut_dir}")
        print(f"📁 Results directory:      {self.results_base_dir}")

        if self.available_duts:
            print(f"\n🔍 Available DUTs:")
            for bitwidth, dut_file in sorted(self.available_duts.items()):
                print(f"   • {bitwidth}-bit: {dut_file.name}")

    def _find_verilog_base_dir(self, verilog_dir: Optional[str]) -> Path:
        """Find base verilog directory containing LLM subdirs"""
        if verilog_dir:
            path = Path(verilog_dir)
            if path.exists():
                return path

        if self.project_root:
            path = self.project_root / "output" / "verilog"
            if path.exists():
                return path

        current = Path.cwd()
        search_paths = [
            current / "output" / "verilog",
            current / "verilog",
            current.parent / "output" / "verilog",
        ]

        for path in search_paths:
            if path.exists():
                return path

        default = current / "output" / "verilog"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _find_dut_dir(self, dut_dir: Optional[str]) -> Path:
        """Find DUT directory"""
        if dut_dir:
            path = Path(dut_dir)
            if path.exists():
                return path

        if self.project_root:
            path = self.project_root / "output" / "dut"
            if path.exists():
                return path

        current = Path.cwd()
        search_paths = [
            current / "output" / "dut",
            current / "dut",
            current.parent / "output" / "dut",
        ]

        for path in search_paths:
            if path.exists():
                return path

        default = current / "output" / "dut"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _setup_results_dir(self, results_dir: Optional[str]) -> Path:
        """Setup results directory"""
        if results_dir:
            path = Path(results_dir)
        elif self.project_root:
            path = self.project_root / "output" / "results"
        else:
            path = self.verilog_base_dir.parent / "results"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scan_available_duts(self) -> Dict[int, Path]:
        """
        Scan DUT directory for available ALU files.

        Returns:
            Dictionary mapping bitwidth to DUT file path
        """
        duts = {}

        # Look for alu_<bitwidth>bit.v files
        for v_file in self.dut_dir.glob("alu_*bit.v"):
            # Extract bitwidth from filename
            match = re.search(r'alu_(\d+)bit\.v', v_file.name)
            if match:
                bitwidth = int(match.group(1))
                duts[bitwidth] = v_file

        return duts

    def _extract_bitwidth_from_testbench(self, tb_file: Path) -> Optional[int]:
        """
        Extract bitwidth from testbench filename.

        Args:
            tb_file: Testbench file path

        Returns:
            Bitwidth as integer, or None if not found
        """
        # Pattern: alu_16bit_20251209_132949_tb.v
        match = re.search(r'alu_(\d+)bit', tb_file.name)
        if match:
            return int(match.group(1))

        # Also check inside the file if not in filename
        try:
            with open(tb_file, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
                match = re.search(r'(\d+)[-_]bit', content, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        except:
            pass

        return None

    def _find_dut_for_bitwidth(self, bitwidth: int) -> Optional[Path]:
        """
        Find DUT file for given bitwidth.

        Args:
            bitwidth: ALU bitwidth

        Returns:
            Path to DUT file or None
        """
        return self.available_duts.get(bitwidth)

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
        """Check if required simulation tools are installed"""
        print("\n🔧 Checking simulation tools...")

        tools = {}

        # Check iverilog
        try:
            result = subprocess.run(
                ["iverilog", "-V"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            tools["iverilog"] = result.returncode == 0
            if tools["iverilog"]:
                version = result.stderr.split('\n')[0] if result.stderr else "unknown"
                print(f"   ✅ iverilog: {version}")
        except:
            tools["iverilog"] = False
            print(f"   ❌ iverilog: Not found")

        # Check vvp
        try:
            result = subprocess.run(
                ["vvp", "-V"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            tools["vvp"] = result.returncode == 0
            if tools["vvp"]:
                print(f"   ✅ vvp: Available")
        except:
            tools["vvp"] = False
            print(f"   ❌ vvp: Not found")

        # Check gtkwave (optional)
        try:
            subprocess.run(
                ["gtkwave", "--version"],
                capture_output=True,
                timeout=5
            )
            tools["gtkwave"] = True
            print(f"   ✅ gtkwave: Available (optional)")
        except:
            tools["gtkwave"] = False
            print(f"   ⚠️  gtkwave: Not found (optional, for waveform viewing)")

        return tools

    def scan_llm_testbenches(self) -> Dict[str, List[Tuple[Path, int]]]:
        """
        Scan for testbenches organized by LLM provider.

        Returns:
            Dictionary mapping LLM name to list of (testbench_file, bitwidth) tuples
        """
        print(f"\n🔍 Scanning for LLM testbenches...")

        llm_testbenches = {}

        # Check root directory
        for tb in self.verilog_base_dir.glob("*_tb.v"):
            bitwidth = self._extract_bitwidth_from_testbench(tb)
            if bitwidth:
                if "default" not in llm_testbenches:
                    llm_testbenches["default"] = []
                llm_testbenches["default"].append((tb, bitwidth))

        # Check LLM subdirectories
        for subdir in self.verilog_base_dir.iterdir():
            if subdir.is_dir():
                llm_name = subdir.name
                tbs = []
                for tb in subdir.glob("*_tb.v"):
                    bitwidth = self._extract_bitwidth_from_testbench(tb)
                    if bitwidth:
                        tbs.append((tb, bitwidth))

                if tbs:
                    llm_testbenches[llm_name] = tbs

        if not llm_testbenches:
            print(f"   ⚠️  No testbenches found")
            return {}

        print(f"   ✅ Found testbenches from {len(llm_testbenches)} LLM(s):")
        for llm_name, tbs in sorted(llm_testbenches.items()):
            # Group by bitwidth
            by_bitwidth = {}
            for tb, bw in tbs:
                if bw not in by_bitwidth:
                    by_bitwidth[bw] = []
                by_bitwidth[bw].append(tb)

            print(f"      📂 {llm_name}: {len(tbs)} testbench(es)")
            for bw, files in sorted(by_bitwidth.items()):
                dut = self._find_dut_for_bitwidth(bw)
                status = "✅" if dut else "❌"
                print(f"         {status} {bw}-bit: {len(files)} testbench(es)")

        return llm_testbenches

    def compile_and_run(
            self,
            llm_name: str,
            testbench_file: Path,
            bitwidth: int
    ) -> SimulationResult:
        """
        Compile and run simulation for one testbench.

        Args:
            llm_name: Name of LLM provider
            testbench_file: Path to testbench .v file
            bitwidth: ALU bitwidth

        Returns:
            SimulationResult object
        """
        result = SimulationResult(llm_name, testbench_file.stem, bitwidth)

        # Find matching DUT
        dut_file = self._find_dut_for_bitwidth(bitwidth)
        if not dut_file:
            print(f"      ❌ No DUT found for {bitwidth}-bit")
            print(f"      Expected: {self.dut_dir}/alu_{bitwidth}bit.v")
            return result

        result.dut_file = dut_file

        # Create LLM-specific results directory
        llm_results_dir = self.results_base_dir / llm_name
        llm_results_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        vvp_file = llm_results_dir / f"{testbench_file.stem}.vvp"
        log_file = llm_results_dir / "simulation.log"

        # Step 1: Compile
        print(f"\n   📦 Compiling: {testbench_file.name}")
        print(f"      DUT: {dut_file.name}")

        compile_cmd = [
            "iverilog",
            "-g2012",
            "-o", str(vvp_file),
            str(dut_file),
            str(testbench_file),
        ]

        self._debug_print(f"Command: {' '.join(compile_cmd)}", "DEBUG")

        try:
            import time
            start_time = time.time()

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
            )

            result.compile_time = time.time() - start_time

            if compile_result.returncode != 0:
                print(f"      ❌ Compilation failed")
                print(f"      Error: {compile_result.stderr}")
                return result

            print(f"      ✅ Compiled ({result.compile_time:.2f}s)")

        except Exception as e:
            print(f"      ❌ Compilation error: {e}")
            return result

        # Step 2: Run simulation
        print(f"   🚀 Running simulation...")

        try:
            start_time = time.time()

            sim_result = subprocess.run(
                ["vvp", str(vvp_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60,
                cwd=str(llm_results_dir),
            )

            result.sim_time = time.time() - start_time

            if sim_result.returncode == 0:
                result.success = True
                result.simulation_log = sim_result.stdout

                # Parse results from output
                self._parse_simulation_output(result, sim_result.stdout)

                # Save log
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(sim_result.stdout)

                # Find VCD file
                vcd_files = list(llm_results_dir.glob("*.vcd"))
                if vcd_files:
                    result.vcd_file = vcd_files[0]

                print(f"      ✅ Simulation completed ({result.sim_time:.2f}s)")
                print(f"      📊 Results: {result.passed_tests}/{result.total_tests} passed")

            else:
                print(f"      ❌ Simulation failed")
                print(f"      Error: {sim_result.stderr}")

        except Exception as e:
            print(f"      ❌ Simulation error: {e}")

        return result

    def _parse_simulation_output(self, result: SimulationResult, output: str):
        """Parse test statistics from simulation output"""
        total_match = re.search(r'Total:\s*(\d+)', output)
        passed_match = re.search(r'Passed:\s*(\d+)', output)
        failed_match = re.search(r'Failed:\s*(\d+)', output)

        if total_match:
            result.total_tests = int(total_match.group(1))
        if passed_match:
            result.passed_tests = int(passed_match.group(1))
        if failed_match:
            result.failed_tests = int(failed_match.group(1))

        if result.total_tests == 0:
            passed_count = len(re.findall(r'PASSED', output, re.IGNORECASE))
            failed_count = len(re.findall(r'FAILED', output, re.IGNORECASE))
            result.passed_tests = passed_count
            result.failed_tests = failed_count
            result.total_tests = passed_count + failed_count

    def run_all_simulations(self) -> Dict[str, List[SimulationResult]]:
        """
        Run simulations for all LLM providers and all bitwidths.

        Returns:
            Dictionary mapping LLM name to list of results
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Multi-Bitwidth Multi-LLM Simulation Campaign")
        print("=" * 70)

        # Check tools
        tools = self.check_tools()
        if not tools.get("iverilog") or not tools.get("vvp"):
            print("\n❌ Required tools not installed!")
            return {}

        # Scan testbenches
        llm_testbenches = self.scan_llm_testbenches()
        if not llm_testbenches:
            print("\n❌ No testbenches found!")
            return {}

        # Run simulations for each LLM
        all_results = {}

        for llm_name, testbenches in sorted(llm_testbenches.items()):
            print(f"\n{'=' * 70}")
            print(f"📂 LLM Provider: {llm_name}")
            print(f"{'=' * 70}")

            llm_results = []

            for i, (tb_file, bitwidth) in enumerate(testbenches, 1):
                print(f"\n📋 Testbench {i}/{len(testbenches)}: {tb_file.name}")

                result = self.compile_and_run(llm_name, tb_file, bitwidth)
                llm_results.append(result)

                # Save individual result
                result_json = self.results_base_dir / llm_name / f"{tb_file.stem}_result.json"
                with open(result_json, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2)

            all_results[llm_name] = llm_results

        return all_results

    def generate_comparison_report(self, all_results: Dict[str, List[SimulationResult]]):
        """Generate comparison report across all LLMs"""
        report_file = self.results_base_dir / "comparison_report.txt"

        lines = []
        lines.append("=" * 80)
        lines.append(" Multi-Bitwidth Multi-LLM Testbench Quality Comparison Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary table
        lines.append("=" * 80)
        lines.append(" Summary Statistics")
        lines.append("=" * 80)
        lines.append(
            f"{'LLM Provider':<15} {'Bitwidth':<10} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Pass Rate':<10}")
        lines.append("-" * 80)

        for llm_name, results in sorted(all_results.items()):
            # Group by bitwidth
            by_bitwidth = {}
            for r in results:
                if r.bitwidth not in by_bitwidth:
                    by_bitwidth[r.bitwidth] = []
                by_bitwidth[r.bitwidth].append(r)

            for bitwidth, bw_results in sorted(by_bitwidth.items()):
                total_tests = sum(r.total_tests for r in bw_results)
                total_passed = sum(r.passed_tests for r in bw_results)
                total_failed = sum(r.failed_tests for r in bw_results)
                pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

                lines.append(
                    f"{llm_name:<15} {bitwidth:<10} {total_tests:<8} {total_passed:<8} {total_failed:<8} {pass_rate:>9.1f}%")

        lines.append("")

        # Write and display report
        report_content = '\n'.join(lines)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("\n" + report_content)
        print(f"\n📄 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Bitwidth Multi-LLM Simulation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--verilog-dir', help='Base directory containing LLM testbench subdirs')
    parser.add_argument('--dut-dir', help='Directory containing DUT files')
    parser.add_argument('--results-dir', help='Directory to save results')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    runner = SimulationRunner(
        verilog_base_dir=args.verilog_dir,
        dut_dir=args.dut_dir,
        results_dir=args.results_dir,
        project_root=args.project_root,
        debug=args.debug
    )

    all_results = runner.run_all_simulations()

    if all_results:
        runner.generate_comparison_report(all_results)
        print("\n✅ Simulation campaign completed!")
    else:
        print("\n❌ No simulations were run!")
        sys.exit(1)


if __name__ == "__main__":
    main()