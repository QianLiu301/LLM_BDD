"""
Testbench Generator with Quality Analysis - Enhanced Version (Fixed)
===================================================================

FIXED VERSION: All quality scores are now capped at 100%
- Fixed functional coverage calculation
- Fixed input diversity calculation
- Fixed test uniqueness calculation
- Fixed corner case coverage calculation
- Fixed overall quality score calculation

ENHANCED FEATURES:
- ✅ Multi-LLM Support
- ✅ Quality Analysis for each testbench
- ✅ Comprehensive quality metrics
- ✅ Comparison reports across LLMs
- ✅ All scores properly capped at 100%

NEW QUALITY METRICS:
1. Functional Coverage - which operations are tested
2. Input Space Coverage - positive, negative, zero, boundary values
3. Test Uniqueness - duplicate detection
4. Corner Case Coverage - edge cases and overflow scenarios
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict


class NumberFormat(Enum):
    """Number format enumeration"""
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"
    BINARY = "binary"


class TestQualityAnalyzer:
    """
    Analyze testbench quality with comprehensive metrics.
    """

    def __init__(self, bitwidth: int = 16):
        self.bitwidth = bitwidth
        self.max_value = (1 << bitwidth) - 1
        self.min_signed = -(1 << (bitwidth - 1))
        self.max_signed = (1 << (bitwidth - 1)) - 1

    def analyze(self, scenarios: List[Dict], operations: Dict = None) -> Dict:
        """
        Comprehensive quality analysis of test scenarios.

        Returns dictionary with quality metrics
        """
        analysis = {
            'total_tests': len(scenarios),
            'functional_coverage': self._analyze_functional_coverage(scenarios, operations),
            'input_space_coverage': self._analyze_input_space_coverage(scenarios),
            'test_uniqueness': self._analyze_test_uniqueness(scenarios),
            'corner_case_coverage': self._analyze_corner_cases(scenarios),
            'quality_score': 0.0  # Overall quality score (0-100)
        }

        # Calculate overall quality score
        analysis['quality_score'] = self._calculate_quality_score(analysis)

        return analysis

    def _analyze_functional_coverage(self, scenarios: List[Dict], operations: Dict = None) -> Dict:
        """Analyze which operations are covered by tests"""
        op_counts = defaultdict(int)
        unique_ops = set()

        for scenario in scenarios:
            # Try to get opcode from scenario
            op = scenario.get('opcode')
            if op is None:
                # Try alternative field names
                op = scenario.get('operation')

            # Convert to string for consistency
            if isinstance(op, int):
                op = f"{op:04b}"
            elif isinstance(op, str):
                # Ensure it's a 4-bit binary string
                if op.startswith('0x') or op.startswith('0X'):
                    op_int = int(op, 16)
                    op = f"{op_int:04b}"
                elif not all(c in '01' for c in op):
                    # Keep as is for now
                    pass

            if op:
                op_counts[op] += 1
                unique_ops.add(op)

        # Expected operations (ADD, SUB, AND, OR = 4 operations)
        expected_ops = 4
        covered_ops = len(unique_ops)

        # 确保coverage不超过100%
        coverage_percentage = min((covered_ops / expected_ops * 100) if expected_ops > 0 else 0, 100.0)

        return {
            'operations_tested': dict(op_counts),
            'unique_operations': covered_ops,
            'expected_operations': expected_ops,
            'coverage_percentage': coverage_percentage,
            'tests_per_operation': {op: count for op, count in op_counts.items()}
        }

    def _analyze_input_space_coverage(self, scenarios: List[Dict]) -> Dict:
        """Analyze coverage of input space"""
        a_values = []
        b_values = []

        for scenario in scenarios:
            # Try different possible field names
            a = scenario.get('a') or scenario.get('A') or 0
            b = scenario.get('b') or scenario.get('B') or 0

            # Handle negative numbers in signed representation
            if a > self.max_signed:
                a = a - (1 << self.bitwidth)
            if b > self.max_signed:
                b = b - (1 << self.bitwidth)
            a_values.append(a)
            b_values.append(b)

        def categorize_values(values):
            categories = {
                'zero': 0,
                'positive_small': 0,  # 1-100
                'positive_medium': 0,  # 101-1000
                'positive_large': 0,  # > 1000
                'negative_small': 0,  # -1 to -100
                'negative_medium': 0,  # -101 to -1000
                'negative_large': 0,  # < -1000
                'boundary_values': 0  # max, min, near-max, near-min
            }

            for v in values:
                if v == 0:
                    categories['zero'] += 1
                elif v > 0:
                    if v <= 100:
                        categories['positive_small'] += 1
                    elif v <= 1000:
                        categories['positive_medium'] += 1
                    else:
                        categories['positive_large'] += 1
                else:  # v < 0
                    if v >= -100:
                        categories['negative_small'] += 1
                    elif v >= -1000:
                        categories['negative_medium'] += 1
                    else:
                        categories['negative_large'] += 1

                # Check boundary values
                if v in [self.max_signed, self.min_signed,
                        self.max_signed - 1, self.min_signed + 1,
                        self.max_value, 0]:
                    categories['boundary_values'] += 1

            return categories

        a_categories = categorize_values(a_values)
        b_categories = categorize_values(b_values)

        # Calculate diversity score (确保不超过100%)
        total_categories = 8  # Number of categories
        covered_a = sum(1 for v in a_categories.values() if v > 0)
        covered_b = sum(1 for v in b_categories.values() if v > 0)
        diversity_score = min(((covered_a + covered_b) / (2 * total_categories)) * 100, 100.0)

        return {
            'input_a_distribution': a_categories,
            'input_b_distribution': b_categories,
            'diversity_score': diversity_score,
            'has_zero': a_categories['zero'] > 0 or b_categories['zero'] > 0,
            'has_negative': any(k.startswith('negative') for k in a_categories if a_categories[k] > 0) or \
                          any(k.startswith('negative') for k in b_categories if b_categories[k] > 0),
            'has_boundary': a_categories['boundary_values'] > 0 or b_categories['boundary_values'] > 0
        }

    def _analyze_test_uniqueness(self, scenarios: List[Dict]) -> Dict:
        """Analyze test uniqueness and detect duplicates"""
        test_signatures = []
        duplicates = []

        for i, scenario in enumerate(scenarios):
            a = scenario.get('a') or scenario.get('A') or 0
            b = scenario.get('b') or scenario.get('B') or 0
            op = scenario.get('opcode') or scenario.get('operation') or '0000'

            signature = (a, b, str(op))

            if signature in test_signatures:
                # Found duplicate
                dup_index = test_signatures.index(signature)
                duplicates.append({
                    'original_index': dup_index + 1,
                    'duplicate_index': i + 1,
                    'signature': f"a={a}, b={b}, op={op}"
                })
            else:
                test_signatures.append(signature)

        unique_count = len(set(test_signatures))
        total_count = len(scenarios)
        uniqueness_rate = min((unique_count / total_count * 100) if total_count > 0 else 0, 100.0)

        return {
            'total_tests': total_count,
            'unique_tests': unique_count,
            'duplicate_tests': len(duplicates),
            'uniqueness_percentage': uniqueness_rate,
            'duplicates': duplicates[:5]  # Show first 5 duplicates
        }

    def _analyze_corner_cases(self, scenarios: List[Dict]) -> Dict:
        """Analyze coverage of corner cases and edge conditions"""
        corner_cases = {
            'zero_operands': False,
            'max_values': False,
            'min_values': False,
            'overflow_potential': False,
            'underflow_potential': False,
            'sign_boundary': False,
            'all_ones': False,
            'alternating_bits': False,
        }

        corner_case_tests = []

        for scenario in scenarios:
            a = scenario.get('a') or scenario.get('A') or 0
            b = scenario.get('b') or scenario.get('B') or 0
            op = scenario.get('opcode') or scenario.get('operation') or '0000'

            # Convert to signed if needed
            a_signed = a if a <= self.max_signed else a - (1 << self.bitwidth)
            b_signed = b if b <= self.max_signed else b - (1 << self.bitwidth)

            # Check corner cases
            if a == 0 and b == 0:
                corner_cases['zero_operands'] = True
                corner_case_tests.append('zero_operands')

            if a == self.max_value or b == self.max_value:
                corner_cases['all_ones'] = True
                corner_case_tests.append('all_ones')

            if abs(a_signed) == self.max_signed or abs(b_signed) == self.max_signed:
                corner_cases['max_values'] = True
                corner_case_tests.append('max_values')

            if a_signed == self.min_signed or b_signed == self.min_signed:
                corner_cases['min_values'] = True
                corner_case_tests.append('min_values')

            # Check for overflow potential (ADD with large positive numbers)
            if str(op) in ['0000', '0x0'] and a_signed > 0 and b_signed > 0:
                if a_signed + b_signed > self.max_signed:
                    corner_cases['overflow_potential'] = True
                    corner_case_tests.append('overflow_potential')

            # Check for underflow potential (SUB with negative result beyond min)
            if str(op) in ['0001', '0x1'] and a_signed < b_signed:
                if a_signed - b_signed < self.min_signed:
                    corner_cases['underflow_potential'] = True
                    corner_case_tests.append('underflow_potential')

            # Check sign boundary (values near 0x7FFF/0x8000 boundary)
            if abs(a_signed - 0) < 10 or abs(b_signed - 0) < 10:
                corner_cases['sign_boundary'] = True

            # Check alternating bit patterns
            if self.bitwidth == 8:
                if a in [0xAA, 0x55] or b in [0xAA, 0x55]:
                    corner_cases['alternating_bits'] = True
                    corner_case_tests.append('alternating_bits')
            elif self.bitwidth == 16:
                if a in [0xAAAA, 0x5555] or b in [0xAAAA, 0x5555]:
                    corner_cases['alternating_bits'] = True
                    corner_case_tests.append('alternating_bits')

        covered = sum(1 for v in corner_cases.values() if v)
        total = len(corner_cases)

        return {
            'corner_cases_covered': corner_cases,
            'coverage_count': covered,
            'total_corner_cases': total,
            'coverage_percentage': min((covered / total * 100) if total > 0 else 0, 100.0),
            'corner_case_tests': list(set(corner_case_tests))
        }

    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        weights = {
            'functional_coverage': 0.30,  # 30%
            'input_diversity': 0.25,  # 25%
            'uniqueness': 0.20,  # 20%
            'corner_cases': 0.25,  # 25%
        }

        # 确保每个维度的分数不超过100
        scores = {
            'functional_coverage': min(analysis['functional_coverage']['coverage_percentage'], 100.0),
            'input_diversity': min(analysis['input_space_coverage']['diversity_score'], 100.0),
            'uniqueness': min(analysis['test_uniqueness']['uniqueness_percentage'], 100.0),
            'corner_cases': min(analysis['corner_case_coverage']['coverage_percentage'], 100.0),
        }

        total_score = sum(scores[k] * weights[k] for k in weights)

        # 确保总分不超过100
        return round(min(total_score, 100.0), 2)

    def generate_report(self, analysis: Dict, llm_name: str = "Unknown") -> str:
        """Generate human-readable quality report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f" Test Quality Analysis Report - {llm_name}")
        lines.append("=" * 80)
        lines.append("")

        # Overall Score
        score = analysis['quality_score']
        score_emoji = "🎉" if score >= 80 else "✅" if score >= 60 else "⚠️" if score >= 40 else "❌"
        lines.append(f"📊 Overall Quality Score: {score:.1f}/100 {score_emoji}")
        lines.append("")

        # Functional Coverage
        fc = analysis['functional_coverage']
        lines.append("1️⃣  Functional Coverage")
        lines.append("-" * 80)
        lines.append(f"   Operations Tested: {fc['unique_operations']}/{fc['expected_operations']} ({fc['coverage_percentage']:.1f}%)")
        lines.append(f"   Total Tests: {analysis['total_tests']}")
        lines.append("")
        lines.append("   Tests per Operation:")
        for op, count in sorted(fc['tests_per_operation'].items()):
            lines.append(f"      • {op}: {count} tests")
        lines.append("")

        # Input Space Coverage
        isc = analysis['input_space_coverage']
        lines.append("2️⃣  Input Space Coverage")
        lines.append("-" * 80)
        lines.append(f"   Diversity Score: {isc['diversity_score']:.1f}%")
        lines.append(f"   ✓ Zero values: {'Yes' if isc['has_zero'] else 'No'}")
        lines.append(f"   ✓ Negative values: {'Yes' if isc['has_negative'] else 'No'}")
        lines.append(f"   ✓ Boundary values: {'Yes' if isc['has_boundary'] else 'No'}")
        lines.append("")

        # Test Uniqueness
        tu = analysis['test_uniqueness']
        lines.append("3️⃣  Test Uniqueness")
        lines.append("-" * 80)
        lines.append(f"   Unique Tests: {tu['unique_tests']}/{tu['total_tests']} ({tu['uniqueness_percentage']:.1f}%)")
        lines.append(f"   Duplicate Tests: {tu['duplicate_tests']}")
        if tu['duplicates']:
            lines.append("   First few duplicates:")
            for dup in tu['duplicates'][:3]:
                lines.append(f"      • Test {dup['duplicate_index']} duplicates Test {dup['original_index']}: {dup['signature']}")
        lines.append("")

        # Corner Cases
        cc = analysis['corner_case_coverage']
        lines.append("4️⃣  Corner Case Coverage")
        lines.append("-" * 80)
        lines.append(f"   Coverage: {cc['coverage_count']}/{cc['total_corner_cases']} ({cc['coverage_percentage']:.1f}%)")
        lines.append("   Covered corner cases:")
        for case, covered in sorted(cc['corner_cases_covered'].items()):
            emoji = "✅" if covered else "❌"
            lines.append(f"      {emoji} {case.replace('_', ' ').title()}")
        lines.append("")

        # Recommendations
        lines.append("💡 Recommendations")
        lines.append("-" * 80)
        recs = self._generate_recommendations(analysis)
        for i, rec in enumerate(recs, 1):
            lines.append(f"   {i}. {rec}")
        lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []

        # Functional coverage
        fc = analysis['functional_coverage']
        if fc['coverage_percentage'] < 100:
            recommendations.append(
                f"Add tests for missing operations ({fc['expected_operations'] - fc['unique_operations']} operations not covered)"
            )

        op_counts = fc['tests_per_operation']
        if op_counts:
            min_tests = min(op_counts.values())
            if min_tests < 3:
                recommendations.append(
                    f"Some operations have very few tests (minimum: {min_tests}). Consider adding more."
                )

        # Input diversity
        isc = analysis['input_space_coverage']
        if isc['diversity_score'] < 60:
            recommendations.append(
                "Improve input diversity by adding more varied test values"
            )
        if not isc['has_negative']:
            recommendations.append(
                "Add tests with negative numbers to verify signed arithmetic"
            )
        if not isc['has_boundary']:
            recommendations.append(
                "Add boundary value tests (max, min values)"
            )

        # Uniqueness
        tu = analysis['test_uniqueness']
        if tu['duplicate_tests'] > 0:
            recommendations.append(
                f"Remove {tu['duplicate_tests']} duplicate test(s) to improve efficiency"
            )

        # Corner cases
        cc = analysis['corner_case_coverage']
        if cc['coverage_percentage'] < 50:
            uncovered = [k for k, v in cc['corner_cases_covered'].items() if not v]
            recommendations.append(
                f"Add corner case tests: {', '.join(uncovered[:3])}"
            )

        if not recommendations:
            recommendations.append("Excellent test coverage! No major improvements needed.")

        return recommendations


class FeatureParser:
    """Parse .feature files and extract test scenarios"""

    def __init__(self, feature_file: str, debug: bool = True):
        self.feature_file = feature_file
        self.debug = debug
        self.bitwidth = 16
        self.operations = {}
        self.scenarios = []
        self.number_format = NumberFormat.DECIMAL

    def parse(self) -> Dict:
        """Parse a .feature file"""
        with open(self.feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        bitwidth_match = re.search(r'(\d+)[-_]?bit', content, re.IGNORECASE)
        if bitwidth_match:
            self.bitwidth = int(bitwidth_match.group(1))

        self._detect_number_format(content)
        self._extract_operations(content)
        self._extract_scenarios(content)

        # Infer bitwidth from values if not found
        if not bitwidth_match and self.scenarios:
            self.bitwidth = self._infer_bitwidth_from_scenarios()

        return {
            'bitwidth': self.bitwidth,
            'operations': self.operations,
            'scenarios': self.scenarios,
            'number_format': self.number_format.value
        }

    def _infer_bitwidth_from_scenarios(self) -> int:
        """Infer bitwidth from value range"""
        max_value = 0
        for scenario in self.scenarios:
            for key in ['a', 'b', 'result', 'expected_result']:
                if key in scenario:
                    val = scenario[key]
                    if isinstance(val, int) and val > max_value:
                        max_value = val

        # Determine bitwidth from max value
        if max_value <= 255:
            return 8
        elif max_value <= 65535:
            return 16
        elif max_value <= 4294967295:
            return 32
        else:
            return 64

    def _detect_number_format(self, content: str):
        """Detect number format used in feature file"""
        if re.search(r'\b0x[0-9A-Fa-f]+\b', content):
            self.number_format = NumberFormat.HEXADECIMAL
        elif re.search(r'\b0b[01]+\b', content):
            self.number_format = NumberFormat.BINARY
        else:
            self.number_format = NumberFormat.DECIMAL

    def _extract_operations(self, content: str):
        """Extract operation definitions"""
        # Look for opcode definitions in various formats
        opcode_pattern = r'(\w+)\s*(?:operation|opcode)?\s*(?:with\s+)?opcode\s+([0-9a-fA-Fx]+)'

        for match in re.finditer(opcode_pattern, content, re.IGNORECASE):
            op_name = match.group(1).upper()
            opcode = match.group(2)

            # Normalize opcode
            if opcode.startswith('0x') or opcode.startswith('0X'):
                opcode_int = int(opcode, 16)
                opcode = format(opcode_int, '04b')
            elif all(c in '01' for c in opcode):
                opcode = opcode.zfill(4)

            self.operations[op_name] = opcode

    def _extract_scenarios(self, content: str):
        """Extract test scenarios from Examples tables"""
        # Find all Examples sections
        examples_pattern = r'Examples?:\s*\n((?:\s*\|.*\n)+)'

        for match in re.finditer(examples_pattern, content, re.MULTILINE):
            table_text = match.group(1)
            rows = [row.strip() for row in table_text.strip().split('\n')]

            if len(rows) < 2:
                continue

            # Parse header
            header = [col.strip() for col in rows[0].split('|') if col.strip()]

            # Parse data rows
            for row in rows[1:]:
                cols = [col.strip() for col in row.split('|') if col.strip()]

                if len(cols) != len(header):
                    continue

                scenario = {}
                for col_name, col_value in zip(header, cols):
                    parsed_value = self._parse_value(col_value)
                    if parsed_value is not None:
                        scenario[col_name.lower()] = parsed_value
                    else:
                        scenario[col_name.lower()] = col_value

                if scenario:
                    self.scenarios.append(scenario)

    def _parse_value(self, value_str: str) -> Optional[int]:
        """Parse value string to integer"""
        try:
            if value_str.startswith('0b') or value_str.startswith('0B'):
                return int(value_str, 2)
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return int(value_str, 16)
            return int(value_str)
        except (ValueError, AttributeError):
            return None


class TestbenchGenerator:
    """Generate Verilog testbench with quality analysis"""

    def __init__(
        self,
        feature_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        dut_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        dut_module_name: Optional[str] = None,
        debug: bool = True
    ):
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None
        self.dut_module_name = dut_module_name

        self.feature_dir = self._find_feature_dir(feature_dir)
        self.output_base_dir = self._setup_output_base_dir(output_dir)
        self.dut_dir = self._find_dut_dir(dut_dir)
        self.quality_reports_dir = self.output_base_dir.parent / "quality_reports"
        self.quality_reports_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Feature directory: {self.feature_dir}")
        print(f"📁 Output base directory: {self.output_base_dir}")
        print(f"📁 DUT directory: {self.dut_dir}")
        print(f"📁 Quality reports: {self.quality_reports_dir}")

    def _find_feature_dir(self, feature_dir: Optional[str]) -> Path:
        """Find .feature files directory"""
        if feature_dir:
            path = Path(feature_dir)
            if path.exists():
                return path

        if self.project_root:
            path = self.project_root / "output" / "bdd"
            if path.exists():
                return path

        current = Path.cwd()
        search_paths = [
            current / "output" / "bdd",
            current / "bdd",
            current.parent / "output" / "bdd",
        ]

        for path in search_paths:
            if path.exists():
                return path

        default = current / "output" / "bdd"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _setup_output_base_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "verilog"
        else:
            path = self.feature_dir.parent / "verilog"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_dut_dir(self, dut_dir: Optional[str]) -> Path:
        """Find DUT directory"""
        if dut_dir:
            path = Path(dut_dir)
            if path.exists():
                return path

        if self.project_root:
            path = self.project_root / "output" / "dut"
        else:
            path = self.feature_dir.parent / "dut"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def scan_features(self) -> List[Tuple[Path, str]]:
        """Scan for .feature files"""
        print(f"\n🔍 Scanning for .feature files...")

        feature_files = []

        for f in self.feature_dir.glob("*.feature"):
            feature_files.append((f, "default"))

        for subdir in self.feature_dir.iterdir():
            if subdir.is_dir():
                llm_name = subdir.name
                for f in subdir.glob("*.feature"):
                    feature_files.append((f, llm_name))

        if not feature_files:
            print(f"   ⚠️  No .feature files found")
            return []

        print(f"   ✅ Found {len(feature_files)} feature file(s)")
        return feature_files

    def _detect_dut_module(self, bitwidth: int = None) -> str:
        """Detect DUT module name"""
        if self.dut_module_name:
            return self.dut_module_name

        if bitwidth:
            return f"alu_{bitwidth}bit"

        alu_files = list(self.dut_dir.glob("alu_*.v"))
        if alu_files:
            return alu_files[0].stem

        return "alu_16bit"

    def generate_testbench(
        self,
        spec: Dict,
        feature_name: str,
        llm_name: str = "default"
    ) -> Tuple[str, Dict]:
        """
        Generate testbench and quality analysis.

        Returns:
            (testbench_content, quality_analysis)
        """
        bitwidth = spec['bitwidth']
        scenarios = spec['scenarios']
        operations = spec.get('operations', {})
        number_format = NumberFormat(spec.get('number_format', 'decimal'))

        # Quality Analysis
        analyzer = TestQualityAnalyzer(bitwidth=bitwidth)
        quality_analysis = analyzer.analyze(scenarios, operations)

        # Detect DUT module
        dut_module = self._detect_dut_module(bitwidth)
        opcode_width = 4

        # Generate testbench content
        lines = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        lines.append("//==============================================================================")
        lines.append(f"// Testbench: {feature_name}")
        lines.append(f"// Generated: {timestamp}")
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// DUT: {dut_module}")
        lines.append(f"// Bitwidth: {bitwidth}")
        lines.append(f"// Test Cases: {len(scenarios)}")
        lines.append(f"// Quality Score: {quality_analysis['quality_score']:.1f}/100")
        lines.append("//==============================================================================")
        lines.append("")
        lines.append(f"module {feature_name}_tb;")
        lines.append("")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Signals")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst;")
        lines.append(f"    reg  [{bitwidth-1}:0] a;")
        lines.append(f"    reg  [{bitwidth-1}:0] b;")
        lines.append(f"    reg  [{opcode_width-1}:0] opcode;")
        lines.append(f"    wire [{bitwidth-1}:0] result;")
        lines.append(f"    wire zero, overflow, negative;")
        lines.append("")
        lines.append("    // Test counters")
        lines.append("    integer passed = 0;")
        lines.append("    integer failed = 0;")
        lines.append("    integer total = 0;")
        lines.append("")

        # DUT instantiation
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Device Under Test")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append(f"    {dut_module} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst(rst),")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .overflow(overflow),")
        lines.append(f"        .negative(negative)")
        lines.append(f"    );")
        lines.append("")

        # Clock generation
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Clock Generation")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append("        clk = 0;")
        lines.append("        forever #5 clk = ~clk;")
        lines.append("    end")
        lines.append("")

        # Test stimulus
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Test Stimulus")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append(f"        $dumpfile(\"{feature_name}_{llm_name}.vcd\");")
        lines.append("        $dumpvars(0, dut);")
        lines.append("")
        lines.append(f"        $display(\"\\n{'='*70}\");")
        lines.append(f"        $display(\"Testbench: {feature_name}\");")
        lines.append(f"        $display(\"LLM Provider: {llm_name}\");")
        lines.append(f"        $display(\"DUT: {dut_module} ({bitwidth}-bit)\");")
        lines.append(f"        $display(\"Test cases: {len(scenarios)}\");")
        lines.append(f"        $display(\"Quality Score: {quality_analysis['quality_score']:.1f}/100\");")
        lines.append(f"        $display(\"{'='*70}\\n\");")
        lines.append("")
        lines.append("        rst = 1;")
        lines.append("        #20 rst = 0;")
        lines.append("        #10;")
        lines.append("")

        # Generate test cases
        lines.append("        //----------------------------------------------------------------------")
        lines.append("        // Test Cases")
        lines.append("        //----------------------------------------------------------------------")

        for i, scenario in enumerate(scenarios, 1):
            # Try different field name variations
            a_val = scenario.get('a') or scenario.get('A') or 0
            b_val = scenario.get('b') or scenario.get('B') or 0
            op_val = scenario.get('opcode') or scenario.get('Opcode') or scenario.get('operation') or '0000'
            expected = scenario.get('expected_result') or scenario.get('Expected_Result') or scenario.get('result') or 0

            if isinstance(op_val, str):
                if op_val.startswith('0x'):
                    op_int = int(op_val, 16)
                    op_val = format(op_int, '04b')
                elif not all(c in '01' for c in op_val):
                    op_val = '0000'

            if number_format == NumberFormat.HEXADECIMAL:
                a_str = f"{bitwidth}'h{a_val:X}"
                b_str = f"{bitwidth}'h{b_val:X}"
                exp_str = f"{bitwidth}'h{expected:X}"
            else:
                a_str = f"{bitwidth}'d{a_val}"
                b_str = f"{bitwidth}'d{b_val}"
                exp_str = f"{bitwidth}'d{expected}"

            lines.append(f"        // Test case {i}")
            lines.append(f"        a = {a_str}; b = {b_str}; opcode = 4'b{op_val};")
            lines.append(f"        #10;")
            lines.append(f"        #10;")
            lines.append(f"        total = total + 1;")
            lines.append(f"        if (result == {exp_str}) begin")
            lines.append(f"            $display(\"✅ Test {i} PASSED: %d op %d = %d\", a, b, result);")
            lines.append(f"            passed = passed + 1;")
            lines.append(f"        end else begin")
            lines.append(f"            $display(\"❌ Test {i} FAILED: %d op %d = %d (expected %d)\", a, b, result, {expected});")
            lines.append(f"            failed = failed + 1;")
            lines.append(f"        end")
            lines.append("")

        # Summary
        lines.append("        //----------------------------------------------------------------------")
        lines.append("        // Test Summary")
        lines.append("        //----------------------------------------------------------------------")
        lines.append(f"        $display(\"\\n{'='*70}\");")
        lines.append(f"        $display(\"Test Summary for {llm_name}\");")
        lines.append(f"        $display(\"{'='*70}\");")
        lines.append("        $display(\"Total:  %0d\", total);")
        lines.append("        $display(\"Passed: %0d\", passed);")
        lines.append("        $display(\"Failed: %0d\", failed);")
        lines.append("")
        lines.append("        if (failed == 0) begin")
        lines.append(f"            $display(\"\\n🎉 ALL TESTS PASSED!\");")
        lines.append("        end else begin")
        lines.append(f"            $display(\"\\n⚠️  SOME TESTS FAILED\");")
        lines.append("        end")
        lines.append(f"        $display(\"{'='*70}\\n\");")
        lines.append("")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")

        testbench_content = '\n'.join(lines)

        return testbench_content, quality_analysis

    def generate_all(self) -> Dict[str, List[Path]]:
        """Generate testbenches for all .feature files with quality analysis"""
        print("\n" + "=" * 70)
        print("🚀 Testbench Generator with Quality Analysis")
        print("=" * 70)

        feature_files = self.scan_features()

        if not feature_files:
            print("\n❌ No .feature files found")
            return {}

        generated_by_llm = {}
        quality_by_llm = {}

        for feature_path, llm_name in feature_files:
            try:
                print(f"\n📖 Processing: {llm_name}/{feature_path.name}")

                parser = FeatureParser(str(feature_path), debug=self.debug)
                spec = parser.parse()

                feature_name = feature_path.stem
                tb_content, quality_analysis = self.generate_testbench(spec, feature_name, llm_name)

                # Save testbench
                llm_output_dir = self.output_base_dir / llm_name
                llm_output_dir.mkdir(parents=True, exist_ok=True)

                tb_path = llm_output_dir / f"{feature_name}_tb.v"
                with open(tb_path, 'w', encoding='utf-8') as f:
                    f.write(tb_content)

                print(f"   ✅ Testbench: {tb_path.name}")

                # Save quality report
                analyzer = TestQualityAnalyzer(bitwidth=spec['bitwidth'])
                quality_report = analyzer.generate_report(quality_analysis, llm_name)

                llm_quality_dir = self.quality_reports_dir / llm_name
                llm_quality_dir.mkdir(parents=True, exist_ok=True)

                quality_path = llm_quality_dir / f"{feature_name}_quality.txt"
                with open(quality_path, 'w', encoding='utf-8') as f:
                    f.write(quality_report)

                print(f"   📊 Quality: {quality_analysis['quality_score']:.1f}/100")

                # Track results
                if llm_name not in generated_by_llm:
                    generated_by_llm[llm_name] = []
                    quality_by_llm[llm_name] = []

                generated_by_llm[llm_name].append(tb_path)
                quality_by_llm[llm_name].append(quality_analysis)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # Generate comparison report
        if quality_by_llm:
            self._generate_quality_comparison(quality_by_llm)

        # Summary
        print("\n" + "=" * 70)
        print(f"✨ Generation Complete!")
        print("=" * 70)

        total = sum(len(files) for files in generated_by_llm.values())
        print(f"\n📊 Summary:")
        print(f"   Total testbenches: {total}")
        print(f"   LLM providers: {len(generated_by_llm)}")
        print(f"   Quality reports: {self.quality_reports_dir}")
        print()

        for llm_name, files in sorted(generated_by_llm.items()):
            avg_quality = sum(q['quality_score'] for q in quality_by_llm[llm_name]) / len(quality_by_llm[llm_name])
            print(f"   📂 {llm_name}: {len(files)} testbench(es), Avg Quality: {avg_quality:.1f}/100")

        return generated_by_llm

    def _generate_quality_comparison(self, quality_by_llm: Dict[str, List[Dict]]):
        """Generate comparison report across all LLMs"""
        comparison_path = self.quality_reports_dir / "quality_comparison.txt"

        lines = []
        lines.append("=" * 80)
        lines.append(" Multi-LLM Testbench Quality Comparison")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary table
        lines.append("=" * 80)
        lines.append(" Overall Quality Scores")
        lines.append("=" * 80)
        lines.append(f"{'LLM Provider':<15} {'Testbenches':<12} {'Avg Quality':<12} {'Best':<8} {'Worst':<8}")
        lines.append("-" * 80)

        all_scores = []
        for llm_name, analyses in sorted(quality_by_llm.items()):
            scores = [a['quality_score'] for a in analyses]
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            worst_score = min(scores)

            all_scores.append((llm_name, avg_score))

            lines.append(f"{llm_name:<15} {len(analyses):<12} {avg_score:>10.1f}% {best_score:>6.1f}% {worst_score:>6.1f}%")

        lines.append("")

        # Rankings
        lines.append("=" * 80)
        lines.append(" Rankings")
        lines.append("=" * 80)
        lines.append("")

        all_scores.sort(key=lambda x: x[1], reverse=True)
        lines.append("🏆 By Average Quality Score:")
        for i, (llm, score) in enumerate(all_scores, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            lines.append(f"   {medal} {i}. {llm}: {score:.1f}%")

        lines.append("")
        lines.append("=" * 80)

        report_content = '\n'.join(lines)

        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n📊 Quality comparison: {comparison_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Testbench Generator with Quality Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--feature-dir', help='Directory containing .feature files')
    parser.add_argument('--output-dir', help='Output directory for testbench files')
    parser.add_argument('--dut-dir', help='Directory containing DUT files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--dut-module', help='DUT module name')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    generator = TestbenchGenerator(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        dut_dir=args.dut_dir,
        project_root=args.project_root,
        dut_module_name=args.dut_module,
        debug=args.debug
    )

    generated_by_llm = generator.generate_all()

    if generated_by_llm:
        print("\n📋 NEXT STEPS:")
        print("=" * 70)
        print("1. Review quality reports in: output/quality_reports/")
        print("2. Run simulations: python simulation_runner.py")
        print("=" * 70)


if __name__ == "__main__":
    main()