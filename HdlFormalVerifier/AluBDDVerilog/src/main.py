# #!/usr/bin/env python3
# """
# Dynamic ALU Verification System
# 集成完整工作流：Spec → BDD → Verilog → 仿真 → 波形显示
# 支持实时ALU参数输入、自动化测试、仿真执行和GTKWave波形显示
# """
#
# import tkinter as tk
# from tkinter import ttk, messagebox, filedialog
# import threading
# import time
# import os
# import subprocess
# from pathlib import Path
# import sys
#
#
# class ALUDynamicVerifier:
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("ALU Dynamic Verification System - Integrated Workflow")
#         self.root.geometry("1000x800")
#
#         # 项目根目录
#         self.project_root = Path.cwd()
#
#         # 创建必要的目录
#         self.setup_directories()
#
#         # 初始化工作流管理器
#         self.setup_workflow()
#
#         # 构建UI
#         self.setup_ui()
#
#         # 仿真状态
#         self.is_running = False
#
#     def setup_directories(self):
#         """创建项目目录结构"""
#         dirs = [
#             "verilog",
#             "src",
#             "specs",
#             "output/bdd_spec",
#             "output/bdd",
#             "output/verilog",
#             "output/simulation"
#         ]
#         for dir_name in dirs:
#             (self.project_root / dir_name).mkdir(parents=True, exist_ok=True)
#
#     def setup_workflow(self):
#         """初始化工作流管理器"""
#         try:
#             # 添加src到路径
#             src_path = self.project_root / "src"
#             if str(src_path) not in sys.path:
#                 sys.path.insert(0, str(src_path))
#
#             from workflow import ALUWorkflow
#             self.workflow = ALUWorkflow(self.project_root)
#             self.log_status("✅ 工作流管理器初始化成功")
#         except Exception as e:
#             self.workflow = None
#             print(f"⚠️  工作流管理器初始化失败: {e}")
#             print("    将使用传统仿真模式")
#
#     def setup_ui(self):
#         """构建图形界面"""
#         # 创建笔记本（标签页）
#         notebook = ttk.Notebook(self.root)
#         notebook.pack(fill='both', expand=True, padx=5, pady=5)
#
#         # 标签页1: 动态测试
#         self.dynamic_tab = ttk.Frame(notebook)
#         notebook.add(self.dynamic_tab, text='动态测试')
#         self.setup_dynamic_tab()
#
#         # 标签页2: Spec工作流
#         self.spec_tab = ttk.Frame(notebook)
#         notebook.add(self.spec_tab, text='Spec工作流')
#         self.setup_spec_tab()
#
#     def setup_dynamic_tab(self):
#         """设置动态测试标签页"""
#         main_frame = ttk.Frame(self.dynamic_tab, padding="10")
#         main_frame.pack(fill='both', expand=True)
#
#         # 输入参数区
#         input_frame = ttk.LabelFrame(main_frame, text="ALU 输入参数", padding="10")
#         input_frame.pack(fill='x', pady=5)
#
#         # A值
#         ttk.Label(input_frame, text="A Value (16-bit):").grid(row=0, column=0, sticky=tk.W, padx=5)
#         self.a_var = tk.StringVar(value="0x000F")
#         self.a_entry = ttk.Entry(input_frame, textvariable=self.a_var, width=20)
#         self.a_entry.grid(row=0, column=1, padx=5, pady=2)
#
#         # B值
#         ttk.Label(input_frame, text="B Value (16-bit):").grid(row=1, column=0, sticky=tk.W, padx=5)
#         self.b_var = tk.StringVar(value="0x000A")
#         self.b_entry = ttk.Entry(input_frame, textvariable=self.b_var, width=20)
#         self.b_entry.grid(row=1, column=1, padx=5, pady=2)
#
#         # 操作码
#         ttk.Label(input_frame, text="Operation Code:").grid(row=2, column=0, sticky=tk.W, padx=5)
#         self.opcode_var = tk.StringVar(value="0000")
#         self.opcode_combo = ttk.Combobox(
#             input_frame,
#             textvariable=self.opcode_var,
#             values=[
#                 "0000 (ADD)", "0001 (SUB)", "0010 (AND)",
#                 "0011 (OR)", "0100 (XOR)", "0101 (SHL)",
#                 "0110 (SHR)", "0111 (NOT)"
#             ],
#             width=18
#         )
#         self.opcode_combo.grid(row=2, column=1, padx=5, pady=2)
#
#         # 控制按钮
#         control_frame = ttk.Frame(main_frame)
#         control_frame.pack(pady=10)
#
#         self.run_button = ttk.Button(
#             control_frame,
#             text="🚀 运行动态工作流",
#             command=self.run_dynamic_workflow
#         )
#         self.run_button.grid(row=0, column=0, padx=5)
#
#         self.gtkwave_button = ttk.Button(
#             control_frame,
#             text="📈 打开波形",
#             command=self.open_gtkwave
#         )
#         self.gtkwave_button.grid(row=0, column=1, padx=5)
#
#         # 状态日志
#         status_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
#         status_frame.pack(fill='both', expand=True, pady=5)
#
#         # 文本框和滚动条
#         text_scroll_frame = ttk.Frame(status_frame)
#         text_scroll_frame.pack(fill='both', expand=True)
#
#         self.status_text = tk.Text(text_scroll_frame, height=15, width=80, wrap=tk.WORD)
#         scrollbar = ttk.Scrollbar(text_scroll_frame, orient="vertical",
#                                   command=self.status_text.yview)
#         self.status_text.configure(yscrollcommand=scrollbar.set)
#         self.status_text.grid(row=0, column=0, sticky='nsew')
#         scrollbar.grid(row=0, column=1, sticky='ns')
#
#         text_scroll_frame.columnconfigure(0, weight=1)
#         text_scroll_frame.rowconfigure(0, weight=1)
#
#         # 预期结果
#         result_frame = ttk.LabelFrame(main_frame, text="预期结果", padding="10")
#         result_frame.pack(fill='x', pady=5)
#
#         self.result_var = tk.StringVar(value="等待计算...")
#         result_label = ttk.Label(result_frame, textvariable=self.result_var,
#                                  font=('Arial', 10, 'bold'))
#         result_label.pack()
#
#     def setup_spec_tab(self):
#         """设置Spec工作流标签页"""
#         main_frame = ttk.Frame(self.spec_tab, padding="10")
#         main_frame.pack(fill='both', expand=True)
#
#         # Spec文件选择
#         file_frame = ttk.LabelFrame(main_frame, text="Spec 文件", padding="10")
#         file_frame.pack(fill='x', pady=5)
#
#         ttk.Label(file_frame, text="Spec文件路径:").grid(row=0, column=0, sticky=tk.W, padx=5)
#         self.spec_file_var = tk.StringVar(value="")
#         self.spec_entry = ttk.Entry(file_frame, textvariable=self.spec_file_var, width=50)
#         self.spec_entry.grid(row=0, column=1, padx=5, pady=2)
#
#         self.browse_button = ttk.Button(file_frame, text="浏览...",
#                                         command=self.browse_spec_file)
#         self.browse_button.grid(row=0, column=2, padx=5)
#
#         # 工作流选项
#         options_frame = ttk.LabelFrame(main_frame, text="工作流选项", padding="10")
#         options_frame.pack(fill='x', pady=5)
#
#         self.gen_bdd_var = tk.BooleanVar(value=True)
#         ttk.Checkbutton(options_frame, text="生成BDD测试",
#                         variable=self.gen_bdd_var).pack(anchor=tk.W)
#
#         self.gen_verilog_var = tk.BooleanVar(value=True)
#         ttk.Checkbutton(options_frame, text="生成Verilog代码",
#                         variable=self.gen_verilog_var).pack(anchor=tk.W)
#
#         self.run_sim_var = tk.BooleanVar(value=False)
#         ttk.Checkbutton(options_frame, text="运行仿真（需要提供测试参数）",
#                         variable=self.run_sim_var).pack(anchor=tk.W)
#
#         # 仿真参数（可选）
#         sim_frame = ttk.LabelFrame(main_frame, text="仿真参数（可选）", padding="10")
#         sim_frame.pack(fill='x', pady=5)
#
#         ttk.Label(sim_frame, text="A值:").grid(row=0, column=0, sticky=tk.W, padx=5)
#         self.spec_a_var = tk.StringVar(value="")
#         ttk.Entry(sim_frame, textvariable=self.spec_a_var, width=15).grid(
#             row=0, column=1, padx=5)
#
#         ttk.Label(sim_frame, text="B值:").grid(row=0, column=2, sticky=tk.W, padx=5)
#         self.spec_b_var = tk.StringVar(value="")
#         ttk.Entry(sim_frame, textvariable=self.spec_b_var, width=15).grid(
#             row=0, column=3, padx=5)
#
#         ttk.Label(sim_frame, text="OpCode:").grid(row=1, column=0, sticky=tk.W, padx=5)
#         self.spec_opcode_var = tk.StringVar(value="")
#         ttk.Entry(sim_frame, textvariable=self.spec_opcode_var, width=15).grid(
#             row=1, column=1, padx=5)
#
#         # 执行按钮
#         control_frame = ttk.Frame(main_frame)
#         control_frame.pack(pady=10)
#
#         self.run_workflow_button = ttk.Button(
#             control_frame,
#             text="🚀 运行完整工作流",
#             command=self.run_spec_workflow
#         )
#         self.run_workflow_button.pack()
#
#         # 状态日志
#         status_frame = ttk.LabelFrame(main_frame, text="工作流日志", padding="10")
#         status_frame.pack(fill='both', expand=True, pady=5)
#
#         text_scroll_frame = ttk.Frame(status_frame)
#         text_scroll_frame.pack(fill='both', expand=True)
#
#         self.spec_status_text = tk.Text(text_scroll_frame, height=15, width=80, wrap=tk.WORD)
#         scrollbar2 = ttk.Scrollbar(text_scroll_frame, orient="vertical",
#                                    command=self.spec_status_text.yview)
#         self.spec_status_text.configure(yscrollcommand=scrollbar2.set)
#         self.spec_status_text.grid(row=0, column=0, sticky='nsew')
#         scrollbar2.grid(row=0, column=1, sticky='ns')
#
#         text_scroll_frame.columnconfigure(0, weight=1)
#         text_scroll_frame.rowconfigure(0, weight=1)
#
#     def log_status(self, message, tab='dynamic'):
#         """在日志区打印消息"""
#         timestamp = time.strftime("%H:%M:%S")
#         text_widget = self.status_text if tab == 'dynamic' else self.spec_status_text
#         text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
#         text_widget.see(tk.END)
#         self.root.update()
#
#     def browse_spec_file(self):
#         """浏览并选择spec文件"""
#         filename = filedialog.askopenfilename(
#             title="选择Spec文件",
#             initialdir=self.project_root / "specs",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
#         )
#         if filename:
#             self.spec_file_var.set(filename)
#
#     def parse_input_values(self):
#         """解析输入字段"""
#         try:
#             a_val = int(self.a_var.get(), 0) & 0xFFFF
#             b_val = int(self.b_var.get(), 0) & 0xFFFF
#             opcode_str = self.opcode_var.get().split()[0] if ' ' in self.opcode_var.get() else self.opcode_var.get()
#             opcode = int(opcode_str, 2) & 0xF
#             return a_val, b_val, opcode
#         except ValueError as e:
#             self.log_status(f"❌ 输入格式错误: {e}")
#             messagebox.showerror("输入错误", f"输入格式无效: {e}")
#             return None, None, None
#
#     def calculate_expected_result(self, a, b, opcode):
#         """计算预期的ALU结果"""
#         operations = {
#             0: lambda x, y: x + y,  # ADD
#             1: lambda x, y: x - y,  # SUB
#             2: lambda x, y: x & y,  # AND
#             3: lambda x, y: x | y,  # OR
#             4: lambda x, y: x ^ y,  # XOR
#             5: lambda x, y: x << 1,  # SHL
#             6: lambda x, y: x >> 1,  # SHR
#             7: lambda x, y: ~x  # NOT
#         }
#
#         if opcode in operations:
#             result = operations[opcode](a, b) & 0xFFFF
#             return result
#         return 0
#
#     def run_dynamic_workflow(self):
#         """运行动态工作流"""
#         if self.is_running:
#             self.log_status("⚠️  工作流正在运行，请等待...")
#             return
#
#         if self.workflow is None:
#             self.log_status("❌ 工作流管理器未初始化")
#             messagebox.showerror("错误", "工作流管理器未初始化")
#             return
#
#         a, b, opcode = self.parse_input_values()
#         if a is None:
#             return
#
#         expected = self.calculate_expected_result(a, b, opcode)
#         self.result_var.set(
#             f"A=0x{a:04X}, B=0x{b:04X}, OpCode={opcode:04b}, "
#             f"预期结果=0x{expected:04X}"
#         )
#
#         self.log_status(f"🚀 启动动态工作流: A=0x{a:04X}, B=0x{b:04X}, OpCode={opcode:04b}")
#
#         thread = threading.Thread(
#             target=self._run_dynamic_workflow_thread,
#             args=(a, b, opcode)
#         )
#         thread.daemon = True
#         thread.start()
#
#     def _run_dynamic_workflow_thread(self, a, b, opcode):
#         """后台动态工作流线程"""
#         self.is_running = True
#         self.run_button.config(state='disabled')
#
#         try:
#             # 运行动态工作流
#             success = self.workflow.run_dynamic_workflow(
#                 a_value=a,
#                 b_value=b,
#                 opcode=opcode,
#                 open_wave=False
#             )
#
#             if success:
#                 self.log_status("✅ 动态工作流执行成功！")
#                 vcd_path = self.workflow.get_vcd_file_path()
#                 if vcd_path:
#                     self.log_status(f"📈 波形文件: {vcd_path}")
#             else:
#                 self.log_status("❌ 动态工作流执行失败")
#
#         except Exception as e:
#             self.log_status(f"❌ 执行错误: {str(e)}")
#             import traceback
#             self.log_status(f"详细信息:\n{traceback.format_exc()}")
#         finally:
#             self.is_running = False
#             self.run_button.config(state='normal')
#
#     def run_spec_workflow(self):
#         """运行Spec工作流"""
#         if self.is_running:
#             self.log_status("⚠️  工作流正在运行，请等待...", 'spec')
#             return
#
#         if self.workflow is None:
#             self.log_status("❌ 工作流管理器未初始化", 'spec')
#             messagebox.showerror("错误", "工作流管理器未初始化")
#             return
#
#         spec_file = self.spec_file_var.get()
#         if not spec_file or not Path(spec_file).exists():
#             self.log_status("❌ 请选择有效的Spec文件", 'spec')
#             messagebox.showerror("错误", "请选择有效的Spec文件")
#             return
#
#         self.log_status(f"🚀 启动完整工作流: {spec_file}", 'spec')
#
#         # 解析仿真参数（如果提供）
#         sim_params = None
#         if self.run_sim_var.get():
#             try:
#                 a = int(self.spec_a_var.get(), 0) if self.spec_a_var.get() else None
#                 b = int(self.spec_b_var.get(), 0) if self.spec_b_var.get() else None
#                 op = int(self.spec_opcode_var.get()) if self.spec_opcode_var.get() else None
#
#                 if all(v is not None for v in [a, b, op]):
#                     sim_params = (a, b, op)
#             except ValueError:
#                 self.log_status("⚠️  仿真参数格式错误，将跳过仿真", 'spec')
#
#         thread = threading.Thread(
#             target=self._run_spec_workflow_thread,
#             args=(spec_file, sim_params)
#         )
#         thread.daemon = True
#         thread.start()
#
#     def _run_spec_workflow_thread(self, spec_file, sim_params):
#         """后台Spec工作流线程"""
#         self.is_running = True
#         self.run_workflow_button.config(state='disabled')
#
#         try:
#             # 运行完整工作流
#             a, b, op = sim_params if sim_params else (None, None, None)
#
#             results = self.workflow.run_full_workflow_from_spec(
#                 spec_file=spec_file,
#                 a_value=a,
#                 b_value=b,
#                 opcode=op,
#                 open_wave=False
#             )
#
#             # 输出结果
#             self.log_status("", 'spec')
#             self.log_status("=" * 60, 'spec')
#             self.log_status("📊 工作流执行完成", 'spec')
#             self.log_status(f"  Spec → BDD:     {'✅ 成功' if results['spec_to_bdd'] else '❌ 失败'}", 'spec')
#             self.log_status(f"  Spec → Verilog: {'✅ 成功' if results['verilog_gen'] else '⊘ 跳过'}", 'spec')
#             self.log_status(f"  仿真执行:       {'✅ 成功' if results['simulation'] else '⊘ 跳过'}", 'spec')
#             self.log_status("=" * 60, 'spec')
#
#         except Exception as e:
#             self.log_status(f"❌ 执行错误: {str(e)}", 'spec')
#             import traceback
#             self.log_status(f"详细信息:\n{traceback.format_exc()}", 'spec')
#         finally:
#             self.is_running = False
#             self.run_workflow_button.config(state='normal')
#
#     def open_gtkwave(self):
#         """打开GTKWave查看波形"""
#         if self.workflow:
#             vcd_file = self.workflow.get_vcd_file_path()
#         else:
#             # 传统方式查找VCD文件
#             vcd_paths = [
#                 self.project_root / "output" / "verilog" / "alu_wave.vcd",
#                 self.project_root / "output" / "alu_wave.vcd",
#                 self.project_root / "verilog" / "alu_wave.vcd"
#             ]
#             vcd_file = None
#             for path in vcd_paths:
#                 if path.exists():
#                     vcd_file = path
#                     break
#
#         if vcd_file is None:
#             self.log_status("❌ 未找到VCD文件，请先运行仿真")
#             messagebox.showwarning("文件缺失", "未找到波形文件，请先运行仿真")
#             return
#
#         self.log_status(f"📈 打开波形: {vcd_file}")
#
#         try:
#             if sys.platform == 'win32':
#                 subprocess.Popen(['gtkwave', str(vcd_file)])
#             else:
#                 subprocess.Popen(
#                     ['gtkwave', str(vcd_file)],
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL
#                 )
#             self.log_status("✅ GTKWave已启动")
#         except FileNotFoundError:
#             self.log_status("❌ 未找到GTKWave")
#             messagebox.showerror("工具缺失", "GTKWave未安装或不在PATH中")
#         except Exception as e:
#             self.log_status(f"❌ 启动GTKWave时出错: {e}")
#
#     def run(self):
#         """启动主应用"""
#         self.log_status("=" * 60)
#         self.log_status("🎉 ALU动态验证系统已启动")
#         self.log_status(f"📁 项目目录: {self.project_root}")
#         self.log_status("=" * 60)
#
#         if self.workflow:
#             self.log_status("✅ 工作流管理器: 已启用")
#             self.log_status("   支持: Spec→BDD→Verilog→仿真→波形")
#         else:
#             self.log_status("⚠️  工作流管理器: 未启用")
#             self.log_status("   仅支持: 传统仿真模式")
#
#         self.log_status("=" * 60)
#         self.log_status("")
#         self.log_status("💡 使用说明:")
#         self.log_status("   【动态测试】标签页:")
#         self.log_status("   1. 输入A、B值和操作码")
#         self.log_status("   2. 点击'运行动态工作流'")
#         self.log_status("   3. 完成后点击'打开波形'查看结果")
#         self.log_status("")
#         self.log_status("   【Spec工作流】标签页:")
#         self.log_status("   1. 选择Spec文件")
#         self.log_status("   2. 配置工作流选项")
#         self.log_status("   3. 可选填写仿真参数")
#         self.log_status("   4. 点击'运行完整工作流'")
#         self.log_status("=" * 60)
#
#         self.root.mainloop()
#
#
# def main():
#     print("🚀 启动ALU动态验证系统...")
#     print(f"📁 工作目录: {Path.cwd()}")
#     print("=" * 60)
#
#     app = ALUDynamicVerifier()
#     app.run()
#
#
# if __name__ == "__main__":
#     main()