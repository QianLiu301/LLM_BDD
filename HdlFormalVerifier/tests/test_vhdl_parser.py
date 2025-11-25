"""
VHDL解析器测试
"""
import os
import sys

# 添加项目根目录到路径（当直接运行此文件时需要）
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

from hdl_parser.vhdl_parser import VHDLParser


def test_simple_entity():
    """测试简单entity解析"""
    code = """
entity test is
    port (
        a : in std_logic;
        b : out std_logic
    );
end entity;

architecture behavioral of test is
begin
    b <= a;
end architecture;
"""

    parser = VHDLParser()
    design = parser.parse(code)

    assert design.entity.name == "test"
    assert len(design.entity.ports) == 2
    assert design.entity.ports[0].name == "a"
    assert design.entity.ports[1].name == "b"

    print("✓ test_simple_entity passed!")


def test_multi_bit_ports():
    """测试多位端口解析"""
    code = """
entity alu is
    port (
        a, b : in std_logic_vector(7 downto 0);
        result : out std_logic_vector(7 downto 0)
    );
end entity;

architecture behavioral of alu is
begin
    result <= a;
end architecture;
"""

    parser = VHDLParser()
    design = parser.parse(code)

    assert design.entity.name == "alu"
    assert len(design.entity.ports) == 3  # a, b, result

    # 检查端口位宽
    a_port = design.entity.ports[0]
    assert a_port.name == "a"
    assert a_port.width() == 8

    print("✓ test_multi_bit_ports passed!")


def test_simple_adder():
    """测试简单加法器"""
    code = """
entity simple_adder is
    port (
        a, b, cin : in std_logic;
        sum, cout : out std_logic
    );
end entity;

architecture behavioral of simple_adder is
begin
    sum <= a;
    cout <= b;
end architecture;
"""

    parser = VHDLParser()
    design = parser.parse(code)

    assert design.entity.name == "simple_adder"

    inputs = design.entity.get_inputs()
    outputs = design.entity.get_outputs()

    assert len(inputs) == 3
    assert len(outputs) == 2

    print("✓ test_simple_adder passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Running VHDL Parser Tests")
    print("=" * 70)

    try:
        test_simple_entity()
        test_multi_bit_ports()
        test_simple_adder()

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
