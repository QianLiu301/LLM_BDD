# 8-bit ALU Formal Verification Examples

这个目录包含了8位ALU的完整形式化验证示例，展示了如何使用PSL（Property Specification Language）规范和Z3求解器进行硬件设计验证。

## 文件说明

### 1. 完整版本（时序逻辑）
- **`alu_8bit.vhdl`**: 完整的8位ALU设计，包含时钟和复位逻辑
- **`alu_properties.psl`**: 完整的PSL属性集（25个属性），包括时序属性

**操作码（Opcode）：**
- `000`: ADD - 加法
- `001`: SUB - 减法
- `010`: AND - 按位与
- `011`: OR - 按位或
- `100`: XOR - 按位异或
- `101`: SHL - 左移
- `110`: SHR - 右移
- `111`: CMP - 比较

### 2. 简化版本（组合逻辑）
- **`alu_simple.vhdl`**: 简化的组合逻辑ALU（无时钟）
- **`alu_simple_properties.psl`**: 简化的PSL属性（28个属性）

**操作码（Opcode）：**
- `000`: ADD - 加法
- `001`: SUB - 减法
- `010`: AND - 按位与
- `011`: OR - 按位或
- `100`: XOR - 按位异或
- `101`: NOT - 按位非
- `110`: CMP - 比较

## 验证的属性类型

### 算术属性
- ✓ 加法正确性（无溢出时）
- ✓ 加法进位标志
- ✓ 减法正确性
- ✓ 减法借位标志
- ✓ 与零运算

### 逻辑属性
- ✓ AND/OR/XOR运算正确性
- ✓ 幂等性（A ⊕ A = A）
- ✓ 与零/全1运算
- ✓ 德摩根定律

### 标志位属性
- ✓ 零标志（Zero Flag）正确性
- ✓ 进位标志（Carry Flag）正确性
- ✓ 溢出标志（Overflow Flag）正确性

### 比较属性
- ✓ 相等比较
- ✓ 不等比较

## 如何运行验证

### 方法 1: 在 PyCharm 中运行

1. **配置运行参数**（完整版）：
   ```
   examples/alu_8bit.vhdl examples/alu_properties.psl --verbose
   ```

2. **配置运行参数**（简化版，推荐开始使用）：
   ```
   examples/alu_simple.vhdl examples/alu_simple_properties.psl --verbose
   ```

3. 点击运行按钮 ▶️

### 方法 2: 命令行运行

```bash
# 简化版（推荐）
python main.py examples/alu_simple.vhdl examples/alu_simple_properties.psl --verbose

# 完整版
python main.py examples/alu_8bit.vhdl examples/alu_properties.psl --verbose
```

## 预期输出

```
╔════════════════════════════════════════════════════════╗
║   HDL Formal Verification Tool v0.1.0                  ║
║   VHDL Design + PSL Properties → Z3 Verification       ║
╚════════════════════════════════════════════════════════╝

📖 Parsing VHDL file: examples/alu_simple.vhdl
✓ Successfully parsed entity: alu_simple
  - Inputs: a[8], b[8], opcode[3]
  - Outputs: result[8], zero, carry

📖 Parsing PSL file: examples/alu_simple_properties.psl
✓ Found 28 properties

🔍 Starting Z3 formal verification...
✓ Property 'add_correct': VERIFIED
✓ Property 'add_with_zero': VERIFIED
✓ Property 'sub_correct': VERIFIED
✓ Property 'and_correct': VERIFIED
✓ Property 'xor_self_zero': VERIFIED
...

✅ All properties verified successfully!
```

## Z3 转换说明

本工具将PSL属性转换为Z3 SMT约束：

### PSL 语法 → Z3 约束

```psl
-- PSL Property
property add_correct is
    (opcode = "000" and unsigned(a) + unsigned(b) < 256) ->
    (unsigned(result) = unsigned(a) + unsigned(b));
```

转换为：

```python
# Z3 Constraint
s.add(Implies(
    And(opcode == BV("000", 3), 
        a + b < 256),
    result == a + b
))
```

## 属性验证示例

### 示例 1: 加法正确性

**属性**: 当没有溢出时，加法结果正确

```psl
property add_correct is
    (opcode = "000" and unsigned(a) + unsigned(b) < 256) ->
    (unsigned(result) = unsigned(a) + unsigned(b));
```

**测试用例**:
- `a = 10`, `b = 20`, `opcode = 000` → `result = 30` ✓
- `a = 100`, `b = 150`, `opcode = 000` → `result = 250` ✓

### 示例 2: XOR幂等性

**属性**: A XOR A = 0

```psl
property xor_self_zero is
    (opcode = "100" and a = b) ->
    (result = "00000000");
```

**测试用例**:
- `a = 42`, `b = 42`, `opcode = 100` → `result = 0` ✓

### 示例 3: 德摩根定律

**属性**: NOT(A AND B) = (NOT A) OR (NOT B)

```psl
property demorgan_1 is
    (opcode = "010") ->
    (not (a and b) = (not a) or (not b));
```

## 扩展建议

### 添加更多属性
1. 边界值测试（0x00, 0xFF）
2. 溢出检测完整性
3. 时序依赖属性（仅完整版）
4. 流水线属性（如果扩展设计）

### 性能优化
1. 分批验证属性
2. 使用Z3的增量求解
3. 属性优先级排序

## 参考资料

- **PSL Standard**: IEEE 1850-2010
- **Z3 Documentation**: https://z3prover.github.io/
- **VHDL-93 Standard**: IEEE 1076-1993

## 故障排除

### 问题 1: 属性验证失败
- 检查操作码是否正确
- 验证位宽是否匹配
- 查看详细输出（使用 `--verbose`）

### 问题 2: Z3 超时
- 简化属性条件
- 减少验证的位宽
- 使用简化版本

### 问题 3: 解析错误
- 检查VHDL语法
- 确认PSL属性格式
- 查看错误日志