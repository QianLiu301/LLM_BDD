#!/usr/bin/env python3
"""
GTKWave控制器
用于管理波形文件显示和GTKWave的自动控制
"""

import os
import subprocess
import time
from pathlib import Path
import threading


class GTKWaveController:
    def __init__(self):
        self.gtkwave_proc = None
        self.vcd_file = None
        self.save_file = None
        self.is_monitoring = False

    def create_save_file(self, vcd_path="alu_16bit.vcd"):
        """创建GTKWave保存文件，预设信号显示"""
        save_content = '''[*]
[*] GTKWave Analyzer v3.3.100 (w)1999-2019 BSI
[*] 
@28
[color] 2
alu_16bit_tb.a[15:0]
@29
[color] 3
alu_16bit_tb.b[15:0]
@28
[color] 4
alu_16bit_tb.opcode[3:0]
@29
[color] 5
alu_16bit_tb.result[15:0]
@28
[color] 1
alu_16bit_tb.carry
@29
[color] 1
alu_16bit_tb.zero
@28
[color] 1
alu_16bit_tb.negative
@29
[color] 1
alu_16bit_tb.overflow
@1
[color] 0
+{A}
@28
[color] 2
alu_16bit_tb.uut.a[15:0]
@29
[color] 3
alu_16bit_tb.uut.b[15:0]
@28
[color] 4
alu_16bit_tb.uut.opcode[3:0]
@29
[color] 5
alu_16bit_tb.uut.result[15:0]
@28
[color] 6
alu_16bit_tb.uut.temp_result[16:0]
@29
[color] 1
alu_16bit_tb.uut.carry
@28
[color] 1
alu_16bit_tb.uut.zero
@29
[color] 1
alu_16bit_tb.uut.negative
@28
[color] 1
alu_16bit_tb.uut.overflow
@1
-{A}
[pattern_trace] 1
[pattern_trace] 0
'''

        save_file_path = Path("output") / "alu_display.gtkw"
        with open(save_file_path, 'w') as f:
            f.write(save_content)

        self.save_file = save_file_path
        return save_file_path

    def launch_gtkwave(self, vcd_file="output/alu_16bit.vcd", save_file=None):
        """启动GTKWave并加载VCD文件"""
        vcd_path = Path(vcd_file)

        if not vcd_path.exists():
            print(f"错误: VCD文件不存在 - {vcd_path}")
            return False

        self.vcd_file = vcd_path

        # 如果没有指定保存文件，创建默认的
        if save_file is None:
            save_file = self.create_save_file()

        try:
            # 启动GTKWave
            cmd = ["gtkwave", str(vcd_path)]
            if save_file and Path(save_file).exists():
                cmd.extend(["-S", str(save_file)])

            print(f"启动GTKWave: {' '.join(cmd)}")
            self.gtkwave_proc = subprocess.Popen(cmd)

            print("GTKWave已启动")
            return True

        except FileNotFoundError:
            print("错误: 未找到GTKWave程序，请确认已正确安装")
            return False
        except Exception as e:
            print(f"启动GTKWave时发生错误: {e}")
            return False

    def reload_vcd(self, new_vcd_file=None):
        """重新加载VCD文件（需要GTKWave支持）"""
        if new_vcd_file:
            self.vcd_file = Path(new_vcd_file)

        # 注意: GTKWave没有直接的重新加载命令
        # 这里我们关闭旧的实例并启动新的
        if self.gtkwave_proc and self.gtkwave_proc.poll() is None:
            self.gtkwave_proc.terminate()
            time.sleep(0.5)

        return self.launch_gtkwave(self.vcd_file, self.save_file)

    def close_gtkwave(self):
        """关闭GTKWave"""
        if self.gtkwave_proc and self.gtkwave_proc.poll() is None:
            self.gtkwave_proc.terminate()
            print("GTKWave已关闭")

    def is_gtkwave_running(self):
        """检查GTKWave是否在运行"""
        return self.gtkwave_proc and self.gtkwave_proc.poll() is None

    def start_monitoring(self, vcd_file="output/alu_16bit.vcd", check_interval=1.0):
        """开始监控VCD文件变化，自动重新加载"""
        self.is_monitoring = True
        self.vcd_file = Path(vcd_file)

        def monitor_thread():
            last_modified = 0

            while self.is_monitoring:
                try:
                    if self.vcd_file.exists():
                        current_modified = self.vcd_file.stat().st_mtime

                        if current_modified > last_modified:
                            last_modified = current_modified
                            print(f"检测到VCD文件更新: {self.vcd_file}")

                            # 如果GTKWave正在运行，重新加载
                            if self.is_gtkwave_running():
                                print("重新加载GTKWave...")
                                self.reload_vcd()
                            else:
                                print("启动GTKWave...")
                                self.launch_gtkwave(self.vcd_file)

                except Exception as e:
                    print(f"监控文件时出错: {e}")

                time.sleep(check_interval)

        # 启动监控线程
        monitor = threading.Thread(target=monitor_thread)
        monitor.daemon = True
        monitor.start()

        print(f"开始监控VCD文件: {self.vcd_file}")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        print("停止VCD文件监控")

    def create_custom_save_file(self, signals_config):
        """根据配置创建自定义的保存文件"""
        save_content = '''[*]
[*] GTKWave Analyzer v3.3.100 (w)1999-2019 BSI
[*] 
'''

        color_map = {
            'input': 2,  # 绿色
            'output': 5,  # 蓝色
            'wire': 4,  # 红色
            'reg': 3  # 黄色
        }

        for signal in signals_config:
            signal_name = signal['name']
            signal_type = signal.get('type', 'wire')
            signal_format = signal.get('format', '@28')  # @28=十六进制, @29=十进制

            color = color_map.get(signal_type, 0)

            save_content += f"{signal_format}\n"
            save_content += f"[color] {color}\n"
            save_content += f"{signal_name}\n"

        save_file_path = Path("output") / "custom_display.gtkw"
        with open(save_file_path, 'w') as f:
            f.write(save_content)

        return save_file_path

    def create_alu_save_file(self):
        """创建专门用于ALU显示的保存文件"""
        signals_config = [
            {'name': 'alu_16bit_tb.a[15:0]', 'type': 'input', 'format': '@28'},
            {'name': 'alu_16bit_tb.b[15:0]', 'type': 'input', 'format': '@28'},
            {'name': 'alu_16bit_tb.opcode[3:0]', 'type': 'input', 'format': '@28'},
            {'name': 'alu_16bit_tb.result[15:0]', 'type': 'output', 'format': '@28'},
            {'name': 'alu_16bit_tb.carry', 'type': 'output', 'format': '@28'},
            {'name': 'alu_16bit_tb.zero', 'type': 'output', 'format': '@28'},
            {'name': 'alu_16bit_tb.negative', 'type': 'output', 'format': '@28'},
            {'name': 'alu_16bit_tb.overflow', 'type': 'output', 'format': '@28'},
        ]

        return self.create_custom_save_file(signals_config)

    def generate_tcl_script(self, vcd_file, output_image="waveform.png"):
        """生成TCL脚本用于自动化GTKWave操作"""
        tcl_content = f'''# GTKWave TCL自动化脚本
gtkwave::loadFile {vcd_file}

# 添加信号
gtkwave::addSignalsFromList [list \\
    alu_16bit_tb.a \\
    alu_16bit_tb.b \\
    alu_16bit_tb.opcode \\
    alu_16bit_tb.result \\
    alu_16bit_tb.carry \\
    alu_16bit_tb.zero \\
    alu_16bit_tb.negative \\
    alu_16bit_tb.overflow \\
]

# 设置时间范围
gtkwave::setFromTime 0
gtkwave::setToTime 100

# 缩放到适合窗口
gtkwave::zoomFit

# 保存波形图片
gtkwave::writeVCD {output_image}

# 退出
exit
'''

        tcl_file = Path("output") / "gtkwave_auto.tcl"
        with open(tcl_file, 'w') as f:
            f.write(tcl_content)

        return tcl_file

    def capture_waveform_image(self, vcd_file="output/alu_16bit.vcd",
                               output_image="output/waveform.png"):
        """捕获波形图片"""
        try:
            # 生成TCL脚本
            tcl_script = self.generate_tcl_script(vcd_file, output_image)

            # 使用GTKWave批处理模式
            cmd = ["gtkwave", "-T", str(tcl_script), str(vcd_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"波形图已保存到: {output_image}")
                return True
            else:
                print(f"捕获波形图失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"捕获波形图时出错: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    controller = GTKWaveController()

    # 示例1: 启动GTKWave查看VCD文件
    if os.path.exists("output/alu_16bit.vcd"):
        controller.launch_gtkwave("output/alu_16bit.vcd")

    # 示例2: 开始监控VCD文件变化
    # controller.start_monitoring("output/alu_16bit.vcd")

    # 示例3: 捕获波形图片
    # controller.capture_waveform_image()