import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                             QLineEdit, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import socket
from datetime import datetime
import struct


class RobotControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.socket = None
        self.is_connected = False
        
        self.commands = {
            '홈 위치 정렬': '30 30 00 00 00 22 01 06 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '이동 상태 조회': '30 30 00 00 00 22 01 06 00 2D 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '정지 명령': '30 30 00 00 00 22 01 06 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '흡착ON_1': '30 30 00 00 00 22 01 06 00 0C E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '흡착ON_2': '30 30 00 00 00 22 01 06 00 0C 00 00 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '흡착OFF_1': '30 30 00 00 00 22 01 06 00 0C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '흡착OFF_2': '30 30 00 00 00 22 01 06 00 0C E8 03 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '초기화_1': '30 30 00 00 00 22 01 06 00 0C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00',
            '초기화_2': '30 30 00 00 00 22 01 06 00 0C E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'
        }
        
        self.preset_positions = {
            1: {'x': 100, 'y': 520, 'z': -627, 'r': 0, 'speed': 5000},
            2: {'x': 0, 'y': -520, 'z': -627, 'r': 0, 'speed': 5000}
        }
        
        self.init_ui()
    
    def coordinate_to_bytes(self, value):
        scaled_value = int(value * 1000)
        return struct.pack('<i', scaled_value)
    
    def create_move_command(self, x, y, z, r, is_linear=False, speed=1000, h_value=50):
        if is_linear:
            header = bytes([0x30, 0x30, 0x00, 0x00, 0x00, 0x22, 0x01, 0x06, 0x00, 0x17])
        else:
            header = bytes([0x30, 0x30, 0x00, 0x00, 0x00, 0x22, 0x01, 0x06, 0x00, 0x03])
        
        x_bytes = self.coordinate_to_bytes(x)
        y_bytes = self.coordinate_to_bytes(y)
        z_bytes = self.coordinate_to_bytes(z)
        r_bytes = self.coordinate_to_bytes(r)
        
        speed_bytes = struct.pack('<i', speed)
        
        if is_linear:
            h_bytes = struct.pack('<i', h_value)
            tail = speed_bytes + h_bytes
        else:
            tail = speed_bytes + bytes([0x00, 0x00, 0x00, 0x00])
        
        command = header + x_bytes + y_bytes + z_bytes + r_bytes + tail
        
        return command
        
    def init_ui(self):
        self.setWindowTitle('로봇 TCP/IP 제어 시스템')
        self.setGeometry(100, 100, 1000, 650)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        connection_group = QGroupBox('연결 설정')
        connection_layout = QHBoxLayout()
        
        self.ip_label = QLabel('IP 주소:')
        self.ip_input = QLineEdit('192.168.27.16')
        self.ip_input.setFixedWidth(150)
        
        self.port_label = QLabel('포트:')
        self.port_input = QLineEdit('502')
        self.port_input.setFixedWidth(100)
        
        self.connect_btn = QPushButton('연결')
        self.connect_btn.setFixedWidth(100)
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        self.status_label = QLabel('상태: 연결 안됨')
        
        connection_layout.addWidget(self.ip_label)
        connection_layout.addWidget(self.ip_input)
        connection_layout.addWidget(self.port_label)
        connection_layout.addWidget(self.port_input)
        connection_layout.addWidget(self.connect_btn)
        connection_layout.addWidget(self.status_label)
        connection_layout.addStretch()
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        coordinate_group = QGroupBox('이동 좌표 입력')
        coordinate_layout = QGridLayout()
        
        coordinate_layout.addWidget(QLabel('X 좌표:'), 0, 0)
        self.x_input = QDoubleSpinBox()
        self.x_input.setRange(-10000, 10000)
        self.x_input.setValue(100)
        self.x_input.setDecimals(2)
        self.x_input.setSuffix(' mm')
        self.x_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.x_input, 0, 1)
        
        coordinate_layout.addWidget(QLabel('Y 좌표:'), 0, 2)
        self.y_input = QDoubleSpinBox()
        self.y_input.setRange(-10000, 10000)
        self.y_input.setValue(50)
        self.y_input.setDecimals(2)
        self.y_input.setSuffix(' mm')
        self.y_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.y_input, 0, 3)
        
        coordinate_layout.addWidget(QLabel('Z 좌표:'), 1, 0)
        self.z_input = QDoubleSpinBox()
        self.z_input.setRange(-10000, 10000)
        self.z_input.setValue(-627)
        self.z_input.setDecimals(2)
        self.z_input.setSuffix(' mm')
        self.z_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.z_input, 1, 1)
        
        coordinate_layout.addWidget(QLabel('R 좌표:'), 1, 2)
        self.r_input = QDoubleSpinBox()
        self.r_input.setRange(-360, 360)
        self.r_input.setValue(0.5)
        self.r_input.setDecimals(2)
        self.r_input.setSuffix(' °')
        self.r_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.r_input, 1, 3)
        
        coordinate_layout.addWidget(QLabel('속도:'), 2, 0)
        self.speed_input = QSpinBox()
        self.speed_input.setRange(0, 100000)
        self.speed_input.setValue(1000)
        self.speed_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.speed_input, 2, 1)
        
        coordinate_layout.addWidget(QLabel('H 값 (선형):'), 2, 2)
        self.h_input = QSpinBox()
        self.h_input.setRange(0, 1000)
        self.h_input.setValue(50)
        self.h_input.setMinimumWidth(120)
        coordinate_layout.addWidget(self.h_input, 2, 3)
        
        self.move_btn = QPushButton('일반 이동 실행')
        self.move_btn.setMinimumHeight(50)
        self.move_btn.clicked.connect(lambda: self.send_coordinate_command(False))
        self.move_btn.setEnabled(False)
        coordinate_layout.addWidget(self.move_btn, 3, 0, 1, 2)
        
        self.linear_move_btn = QPushButton('선형 이동 실행')
        self.linear_move_btn.setMinimumHeight(50)
        self.linear_move_btn.clicked.connect(lambda: self.send_coordinate_command(True))
        self.linear_move_btn.setEnabled(False)
        coordinate_layout.addWidget(self.linear_move_btn, 3, 2, 1, 2)
        
        coordinate_group.setLayout(coordinate_layout)
        main_layout.addWidget(coordinate_group)
        
        control_group = QGroupBox('기타 제어 명령')
        control_layout = QGridLayout()
        
        self.control_buttons = {}
        button_configs = [
            ('홈 위치 정렬', 0, 0),
            ('이동 상태 조회', 0, 1),
            ('정지 명령', 0, 2),
            ('1번 위치로 이동', 1, 0),
            ('2번 위치로 이동', 1, 1),
            ('흡착ON', 2, 0),
            ('흡착OFF', 2, 1),
            ('초기화', 2, 2)
        ]
        
        for name, row, col in button_configs:
            btn = QPushButton(name)
            btn.setMinimumHeight(60)
            
            if '위치로 이동' in name:
                position_num = int(name[0])
                btn.clicked.connect(lambda checked, pos=position_num: self.move_to_preset_position(pos))
            elif name == '흡착ON':
                btn.clicked.connect(self.send_suction_on)
            elif name == '흡착OFF':
                btn.clicked.connect(self.send_suction_off)
            elif name == '초기화':
                btn.clicked.connect(self.send_reset)
            else:
                btn.clicked.connect(lambda checked, cmd_name=name: self.send_command(cmd_name))
            
            btn.setEnabled(False)
            control_layout.addWidget(btn, row, col)
            self.control_buttons[name] = btn
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        send_group = QGroupBox('전송 데이터')
        send_layout = QVBoxLayout()
        self.send_display = QTextEdit()
        self.send_display.setReadOnly(True)
        self.send_display.setMaximumHeight(120)
        self.send_display.setFont(QFont('Courier', 9))
        send_layout.addWidget(self.send_display)
        send_group.setLayout(send_layout)
        main_layout.addWidget(send_group)
        
        receive_group = QGroupBox('수신 데이터')
        receive_layout = QVBoxLayout()
        self.receive_display = QTextEdit()
        self.receive_display.setReadOnly(True)
        self.receive_display.setMaximumHeight(120)
        self.receive_display.setFont(QFont('Courier', 9))
        receive_layout.addWidget(self.receive_display)
        receive_group.setLayout(receive_layout)
        main_layout.addWidget(receive_group)
        
    def toggle_connection(self):
        if not self.is_connected:
            self.connect_to_robot()
        else:
            self.disconnect_from_robot()
            
    def connect_to_robot(self):
        try:
            ip = self.ip_input.text()
            port = int(self.port_input.text())
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((ip, port))
            
            self.is_connected = True
            self.status_label.setText(f'상태: 연결됨 ({ip}:{port})')
            self.connect_btn.setText('연결 해제')
            
            for btn in self.control_buttons.values():
                btn.setEnabled(True)
            
            self.move_btn.setEnabled(True)
            self.linear_move_btn.setEnabled(True)
            
            self.ip_input.setEnabled(False)
            self.port_input.setEnabled(False)
            
        except Exception as e:
            self.status_label.setText(f'상태: 연결 실패 - {str(e)}')
            
    def disconnect_from_robot(self):
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            
            self.is_connected = False
            self.status_label.setText('상태: 연결 안됨')
            self.connect_btn.setText('연결')
            
            for btn in self.control_buttons.values():
                btn.setEnabled(False)
            
            self.move_btn.setEnabled(False)
            self.linear_move_btn.setEnabled(False)
            
            self.ip_input.setEnabled(True)
            self.port_input.setEnabled(True)
            
        except Exception as e:
            self.status_label.setText(f'상태: 연결 해제 오류 - {str(e)}')
    
    def send_coordinate_command(self, is_linear):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            x = self.x_input.value()
            y = self.y_input.value()
            z = self.z_input.value()
            r = self.r_input.value()
            speed = self.speed_input.value()
            
            if is_linear:
                h_value = self.h_input.value()
                byte_data = self.create_move_command(x, y, z, r, is_linear, speed, h_value)
            else:
                byte_data = self.create_move_command(x, y, z, r, is_linear, speed)
            
            self.socket.sendall(byte_data)
            
            hex_string = ' '.join([f'{b:02X}' for b in byte_data])
            command_name = '선형 이동' if is_linear else '일반 이동'
            
            self.send_display.clear()
            self.send_display.append(f'[{command_name}]')
            if is_linear:
                self.send_display.append(f'좌표: X={x}, Y={y}, Z={z}, R={r}, 속도={speed}, H={h_value}')
            else:
                self.send_display.append(f'좌표: X={x}, Y={y}, Z={z}, R={r}, 속도={speed}')
            self.send_display.append(hex_string)
            
            self.socket.settimeout(1.0)
            response = self.socket.recv(1024)
            
            response_hex = ' '.join([f'{b:02X}' for b in response])
            
            self.receive_display.clear()
            self.receive_display.append(f'[응답 - {len(response)} bytes]')
            self.receive_display.append(response_hex)
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            self.receive_display.append('응답 없음')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
    
    def move_to_preset_position(self, position_num):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            pos = self.preset_positions[position_num]
            x = pos['x']
            y = pos['y']
            z = pos['z']
            r = pos['r']
            speed = pos['speed']
            
            byte_data = self.create_move_command(x, y, z, r, False, speed)
            
            self.socket.sendall(byte_data)
            
            hex_string = ' '.join([f'{b:02X}' for b in byte_data])
            
            self.send_display.clear()
            self.send_display.append(f'[{position_num}번 위치로 이동]')
            self.send_display.append(f'좌표: X={x}, Y={y}, Z={z}, R={r}, 속도={speed}')
            self.send_display.append(hex_string)
            
            self.socket.settimeout(1.0)
            response = self.socket.recv(1024)
            
            response_hex = ' '.join([f'{b:02X}' for b in response])
            
            self.receive_display.clear()
            self.receive_display.append(f'[응답 - {len(response)} bytes]')
            self.receive_display.append(response_hex)
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            self.receive_display.append('응답 없음')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
    
    def send_suction_on(self):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            self.send_display.clear()
            self.send_display.append('[흡착ON]')
            
            hex_string1 = self.commands['흡착ON_1']
            hex_values1 = hex_string1.split()
            byte_data1 = bytes([int(x, 16) for x in hex_values1])
            self.socket.sendall(byte_data1)
            self.send_display.append(f'명령 1: {hex_string1}')
            
            self.socket.settimeout(1.0)
            response1 = self.socket.recv(1024)
            
            hex_string2 = self.commands['흡착ON_2']
            hex_values2 = hex_string2.split()
            byte_data2 = bytes([int(x, 16) for x in hex_values2])
            self.socket.sendall(byte_data2)
            self.send_display.append(f'명령 2: {hex_string2}')
            
            self.socket.settimeout(1.0)
            response2 = self.socket.recv(1024)
            
            self.receive_display.clear()
            self.receive_display.append('[응답 1]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response1]))
            self.receive_display.append('[응답 2]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response2]))
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
    
    def send_suction_off(self):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            self.send_display.clear()
            self.send_display.append('[흡착OFF]')
            
            hex_string1 = self.commands['흡착OFF_1']
            hex_values1 = hex_string1.split()
            byte_data1 = bytes([int(x, 16) for x in hex_values1])
            self.socket.sendall(byte_data1)
            self.send_display.append(f'명령 1: {hex_string1}')
            
            self.socket.settimeout(1.0)
            response1 = self.socket.recv(1024)
            
            hex_string2 = self.commands['흡착OFF_2']
            hex_values2 = hex_string2.split()
            byte_data2 = bytes([int(x, 16) for x in hex_values2])
            self.socket.sendall(byte_data2)
            self.send_display.append(f'명령 2: {hex_string2}')
            
            self.socket.settimeout(1.0)
            response2 = self.socket.recv(1024)
            
            self.receive_display.clear()
            self.receive_display.append('[응답 1]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response1]))
            self.receive_display.append('[응답 2]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response2]))
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
    
    def send_reset(self):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            self.send_display.clear()
            self.send_display.append('[초기화]')
            
            hex_string1 = self.commands['초기화_1']
            hex_values1 = hex_string1.split()
            byte_data1 = bytes([int(x, 16) for x in hex_values1])
            self.socket.sendall(byte_data1)
            self.send_display.append(f'명령 1: {hex_string1}')
            
            self.socket.settimeout(1.0)
            response1 = self.socket.recv(1024)
            
            hex_string2 = self.commands['초기화_2']
            hex_values2 = hex_string2.split()
            byte_data2 = bytes([int(x, 16) for x in hex_values2])
            self.socket.sendall(byte_data2)
            self.send_display.append(f'명령 2: {hex_string2}')
            
            self.socket.settimeout(1.0)
            response2 = self.socket.recv(1024)
            
            self.receive_display.clear()
            self.receive_display.append('[응답 1]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response1]))
            self.receive_display.append('[응답 2]')
            self.receive_display.append(' '.join([f'{b:02X}' for b in response2]))
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
            
    def send_command(self, command_name):
        if not self.is_connected or not self.socket:
            self.status_label.setText('상태: 로봇이 연결되지 않음')
            return
        
        try:
            hex_string = self.commands[command_name]
            hex_values = hex_string.split()
            byte_data = bytes([int(x, 16) for x in hex_values])
            
            self.socket.sendall(byte_data)
            
            self.send_display.clear()
            self.send_display.append(f'[{command_name}]')
            self.send_display.append(hex_string)
            
            self.socket.settimeout(1.0)
            response = self.socket.recv(1024)
            
            response_hex = ' '.join([f'{b:02X}' for b in response])
            
            self.receive_display.clear()
            self.receive_display.append(f'[응답 - {len(response)} bytes]')
            self.receive_display.append(response_hex)
            
        except socket.timeout:
            self.receive_display.clear()
            self.receive_display.append('[타임아웃]')
            self.receive_display.append('응답 없음')
            
        except Exception as e:
            self.status_label.setText(f'상태: 통신 오류 - {str(e)}')
            self.disconnect_from_robot()
            
    def closeEvent(self, event):
        if self.is_connected:
            self.disconnect_from_robot()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotControlGUI()
    window.show()
    sys.exit(app.exec_())