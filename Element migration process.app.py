import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import csv
import json
from io import StringIO, BytesIO
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib
import os
import platform

# ===================== 全局配置 =====================
# 深度优化跨平台字体配置（解决中文显示问题）
def setup_chinese_font():
    """跨平台中文字体配置，自动检测可用字体"""
    # 先清空字体缓存
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    # Windows/Linux/macOS 字体优先级
    font_paths = {
        'Windows': ['SimHei', 'Microsoft YaHei', 'FangSong'],
        'Linux': ['WenQuanYi Micro Hei', 'DejaVu Sans'],
        'Darwin': ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
    }
    
    # 检测系统类型
    system = platform.system()
    candidate_fonts = font_paths.get(system, ['DejaVu Sans'])
    
    # 检测可用字体
    available_fonts = [f for f in candidate_fonts if f in plt.rcParams['font.sans-serif']]
    if available_fonts:
        plt.rcParams["font.family"] = available_fonts
    else:
        plt.rcParams["font.family"] = ['DejaVu Sans']
    
    # 基础配置
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

# 初始化字体配置
setup_chinese_font()

st.set_page_config(
    page_title="地球化学元素迁移虚拟仿真平台",
    page_icon="🌍",
    layout="wide"
)

# ===================== 1. 数值模拟核心模块 =====================
class NumericalSimulation:
    """基于有限差分法的元素迁移数值模拟核心类"""

    def __init__(self, domain_size: Tuple[int, int] = (50, 50), dx: float = 1.0, dy: float = 1.0, dt: float = 1.0):
        self.domain_size = domain_size  # 模拟域尺寸 (x, y)
        self.dx, self.dy = dx, dy  # 空间步长
        self.dt = dt  # 时间步长
        self.concentration = np.zeros(domain_size)  # 元素浓度场
        self.time = 0.0  # 当前模拟时间
        self.saturation_concentration = 1.0  # 饱和浓度（用于水-岩反应）
        self.water_mobility = 1.0  # 新增：水的流动性参数（影响Li流失速率）

    def central_difference_x(self, field: np.ndarray) -> np.ndarray:
        """x方向中心差分计算梯度"""
        return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * self.dx)

    def central_difference_y(self, field: np.ndarray) -> np.ndarray:
        """y方向中心差分计算梯度"""
        return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * self.dy)

    def forward_difference_time(self, field: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """时间向前差分更新"""
        return field + rate * self.dt

    def explicit_solver(self, diffusion_coeff: float, reaction_rate: float) -> np.ndarray:
        """显式有限差分求解对流-扩散-反应方程"""
        # 拉普拉斯算子（扩散项）
        laplacian = (
                (np.roll(self.concentration, -1, axis=1) + np.roll(self.concentration, 1,
                                                                   axis=1) - 2 * self.concentration) / self.dx ** 2 +
                (np.roll(self.concentration, -1, axis=0) + np.roll(self.concentration, 1,
                                                                   axis=0) - 2 * self.concentration) / self.dy ** 2
        )
        diffusion_term = diffusion_coeff * laplacian
        reaction_term = -reaction_rate * self.concentration  # 反应项（简化为线性衰减）

        # 更新浓度场
        self.concentration = self.forward_difference_time(self.concentration, diffusion_term + reaction_term)
        self.time += self.dt
        # 确保浓度非负（物理意义约束）
        self.concentration = np.clip(self.concentration, 0, None)
        return self.concentration

    def implicit_solver(self, diffusion_coeff: float, reaction_rate: float, max_iter: int = 10) -> np.ndarray:
        """隐式有限差分求解（Jacobi迭代）- 适配Li流失场景，加入水流动性影响"""
        new_concentration = self.concentration.copy()
        for _ in range(max_iter):
            for i in range(1, self.domain_size[0] - 1):
                for j in range(1, self.domain_size[1] - 1):
                    # 隐式格式离散，加入水流动性系数（放大Li流失速率）
                    mobility_factor = self.water_mobility * 1e-2
                    new_concentration[i, j] = (
                                                      self.concentration[i, j] + self.dt * diffusion_coeff * (
                                                      (self.concentration[i + 1, j] + self.concentration[
                                                          i - 1, j]) / self.dx ** 2 +
                                                      (self.concentration[i, j + 1] + self.concentration[
                                                          i, j - 1]) / self.dy ** 2
                                              ) - mobility_factor * self.concentration[i, j]
                                              ) / (1 + self.dt * (
                            2 * diffusion_coeff * (1 / self.dx ** 2 + 1 / self.dy ** 2) + reaction_rate))
        self.concentration = new_concentration
        self.concentration = np.clip(self.concentration, 0, None)
        self.time += self.dt
        return self.concentration

    def set_water_mobility(self, mobility: float):
        """新增：设置水的流动性参数"""
        self.water_mobility = mobility

    def water_rock_reaction(self, mineral_dissolution_rate: float, surface_area: float) -> float:
        """水-岩相互作用：矿物溶解动力学模型"""
        return mineral_dissolution_rate * surface_area * (1 - self.concentration / self.saturation_concentration)

    def magma_crystallization(self, distribution_coefficient: float, melt_fraction: float) -> np.ndarray:
        """岩浆结晶分异：瑞利结晶模型"""
        return self.concentration * (1 - melt_fraction) ** (distribution_coefficient - 1)

    def reset_concentration(self):
        """重置浓度场"""
        self.concentration = np.zeros(self.domain_size)
        self.time = 0.0
        self.water_mobility = 1.0  # 重置水流动性

# ===================== 2. 场景预设与自定义模块 =====================
class SceneManager:
    """管理内置场景与自定义场景"""

    def __init__(self):
        # 初始化所有场景参数（包含Li场景的water_mobility_range）
        self.scenes: Dict[str, Dict] = {
            "au_hydrothermal": {
                "name": "热液蚀变Au富集",
                "initial_concentration": 0.01,  # ppm
                "temperature_range": (0, 1000),
                "ph_range": (2.0, 8.0),  # 保持原有范围
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "time_range": (100, 20000),
                "dt": 1.0,
                "diffusion_coeff": 1e-6,
                "reaction_rate": 1e-4,
                "solver_type": "explicit"
            },
            "li_weathering": {
                "name": "风化淋滤Li流失",
                "initial_concentration": 50,  # ppm
                "ph_range": (0.0, 12.0),  # PH范围拓展至0-12
                "temperature_range": (0, 1000),
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "water_mobility_range": (0.1, 10.0),  # 水流动性参数范围（必加）
                "time_range": (1000, 100000),
                "dt": 100.0,
                "diffusion_coeff": 1e-7,
                "reaction_rate": 1e-5,
                "solver_type": "implicit"
            }
        }

    def get_scene(self, scene_name: str) -> Dict:
        """安全获取场景参数，返回空字典+默认值避免KeyError"""
        scene = self.scenes.get(scene_name, {})
        # 为Li场景补充默认参数（防止参数缺失）
        if scene_name == "li_weathering":
            scene.setdefault("water_mobility_range", (0.1, 10.0))
            scene.setdefault("ph_range", (0.0, 12.0))
            scene.setdefault("temperature_range", (0, 1000))
            scene.setdefault("initial_concentration", 50.0)
        return scene

    def create_custom_scene(self, name: str, params: Dict) -> Dict:
        """创建自定义场景"""
        self.scenes[name] = params
        return self.scenes[name]

# ===================== 3. 结果可视化与分析模块 =====================
class ResultVisualization:
    """结果可视化与分析工具（修复导出数据格式错误）"""

    def __init__(self, simulation: NumericalSimulation):
        self.simulation = simulation
        setup_chinese_font()

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        """重构等值线图绘制逻辑"""
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor="white")
        fig.suptitle("Concentration Contour Map", fontsize=14, fontweight='bold')

        # 生成浓度等值线（确保梯度可见）
        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        if max_c - min_c < 1e-6:
            levels = np.linspace(min_c, min_c + 5.0, 20)
        else:
            levels = np.linspace(min_c, max_c, 20)

        # 绘制填充等值线
        contour = ax.contourf(
            self.simulation.concentration,
            levels=levels,
            cmap='viridis',
            extend='both',
            alpha=0.8
        )
        ax.contour(
            self.simulation.concentration,
            levels=levels,
            colors='white',
            linewidths=0.5,
            alpha=0.5
        )

        # 添加颜色条
        cbar = fig.colorbar(contour, ax=ax, label='Concentration (ppm)', shrink=0.8)
        cbar.ax.set_ylabel('Concentration (ppm)', fontsize=10)

        # 设置坐标轴
        ax.set_xlabel('Spatial Coordinate X', fontsize=12)
        ax.set_ylabel('Spatial Coordinate Y', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        
        return fig

    def plot_time_series(self, time_points: List[float], concentrations: List[float],
                         title: str = "浓度-时间曲线") -> plt.Figure:
        """绘制浓度随时间变化曲线"""
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
        
        ax.plot(time_points, concentrations, 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Average Concentration (ppm)', fontsize=12)
        ax.set_title("Concentration-Time Curve", fontsize=14, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        
        return fig

    def calculate_enrichment_factor(self, initial_concentration: float) -> float:
        """计算元素富集/流失系数"""
        max_concentration = np.max(self.simulation.concentration)
        factor = max_concentration / initial_concentration if initial_concentration > 0 else 0.0
        # Li流失场景返回流失系数
        if "li_weathering" in st.session_state.get("current_scene", {}).get("name", ""):
            return 1.0 / factor if factor > 0 else 0.0
        return factor

    def export_excel(self) -> bytes:
        """修复Excel导出格式：返回纯bytes（彻底解决Invalid binary data format错误）"""
        try:
            import openpyxl
        except ImportError:
            st.error("缺少Excel依赖：请在终端执行 pip install openpyxl")
            return b""
        
        # 构建数据
        x_coords, y_coords, concs = [], [], []
        nx, ny = self.simulation.domain_size
        for i in range(nx):
            for j in range(ny):
                x_coords.append(i)
                y_coords.append(j)
                concs.append(float(self.simulation.concentration[i, j]))
        
        df = pd.DataFrame({
            'X坐标': x_coords,
            'Y坐标': y_coords,
            '浓度(ppm)': concs
        })
        
        # 核心修复：确保返回纯bytes，而非BytesIO对象
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='浓度数据', index=False)
            output.seek(0)
            excel_bytes = output.getvalue()  # 转为纯字节数据
            output.close()
            return excel_bytes
        except Exception as e:
            st.error(f"Excel导出失败：{str(e)}")
            return b""

    def export_vtk(self) -> str:
        """修复VTK导出格式：返回纯字符串（彻底解决Invalid binary data format错误）"""
        nx, ny = self.simulation.domain_size
        n_points = nx * ny
        
        # 构建VTK内容（纯字符串）
        vtk_content = f"""# vtk DataFile Version 3.0
Geochemical Element Migration Simulation
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS {ny} {nx} 1
ORIGIN 0 0 0
SPACING {self.simulation.dx} {self.simulation.dy} 1
POINT_DATA {n_points}
SCALARS concentration float 1
LOOKUP_TABLE default
"""
        # 写入浓度数据
        for j in range(ny):
            for i in range(nx):
                vtk_content += f"{self.simulation.concentration[i, j]:.6f}\n"
        
        return vtk_content  # 直接返回纯字符串

# ===================== 4. 教学管理模块（保留） =====================
class TeachingManagement:
    """教学任务管理与数据统计"""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.student_data: Dict[str, List[str]] = {}

    def create_task(self, task_id: str, scene_name: str, param_ranges: Dict, deadline: str) -> None:
        self.tasks[task_id] = {
            "scene_name": scene_name,
            "param_ranges": param_ranges,
            "deadline": deadline,
            "submissions": {}
        }

    def submit_experiment(self, task_id: str, student_id: str, params: Dict, results: Dict) -> None:
        if task_id in self.tasks:
            self.tasks[task_id]["submissions"][student_id] = {
                "params": params,
                "results": results,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if student_id not in self.student_data:
                self.student_data[student_id] = []
            self.student_data[student_id].append(task_id)

    def auto_grade(self, task_id: str, student_id: str) -> Tuple[str, str]:
        if task_id not in self.tasks or student_id not in self.tasks[task_id]["submissions"]:
            return "错误", "任务或学生不存在"

        submission = self.tasks[task_id]["submissions"][student_id]
        param_ranges = self.tasks[task_id]["param_ranges"]

        params_valid = True
        for k in param_ranges:
            if k in submission["params"]:
                if not (param_ranges[k][0] <= submission["params"][k] <= param_ranges[k][1]):
                    params_valid = False
                    break

        results_valid = submission["results"]["enrichment_factor"] > 1.0

        if params_valid and results_valid:
            return "通过", "参数设置合理，结果符合预期"
        else:
            return "不通过", "参数超出范围或结果不合理"

    def export_statistics(self, task_id: str) -> Dict:
        if task_id not in self.tasks:
            return {}

        submissions = self.tasks[task_id]["submissions"]
        total_students = len(self.student_data)
        completion_rate = len(submissions) / total_students if total_students > 0 else 0.0
        param_adjustments = [len(s["params"]) for s in submissions.values()]
        avg_param_adjustments = np.mean(param_adjustments) if param_adjustments else 0.0

        return {
            "任务ID": task_id,
            "完成率": f"{completion_rate * 100:.1f}%",
            "平均参数调整次数": f"{avg_param_adjustments:.1f}",
            "提交记录数": len(submissions)
        }

# ===================== 5. 会话状态初始化（核心修复） =====================
def init_session_state():
    """安全初始化所有会话状态，避免未定义错误"""
    # 初始化核心对象
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation(domain_size=(50, 50), dx=1.0, dy=1.0, dt=1.0)
    if "scene_manager" not in st.session_state:
        st.session_state.scene_manager = SceneManager()
    if "teaching_manager" not in st.session_state:
        st.session_state.teaching_manager = TeachingManagement()
        # 初始化教学任务
        st.session_state.teaching_manager.create_task(
            task_id="GEOCHEM_TASK_001",
            scene_name="au_hydrothermal",
            param_ranges={
                "temperature": (0, 1000),
                "ph": (2.0, 8.0),
                "pressure": (10, 1000),
                "eh": (-200, 400),
                "sulfur_content": (0.01, 1.0),
                "chlorine_content": (0.1, 10.0),
                "time_steps": (100, 20000)
            },
            deadline="2024-12-31"
        )
    # 初始化场景和参数（强制赋默认值）
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = {}
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = {}
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "selected_scene_key" not in st.session_state:
        st.session_state.selected_scene_key = "au_hydrothermal"

# ===================== 6. 主界面逻辑（全容错修复） =====================
def main():
    # 优先初始化会话状态
    init_session_state()

    # 页面标题
    st.title("🌍 地球化学元素迁移虚拟仿真平台")
    st.divider()

    # 左侧：场景选择与参数配置
    with st.sidebar:
        st.header("🔧 实验配置")

        # 1. 场景选择（绑定会话状态）
        scene_options = {
            "au_hydrothermal": "热液蚀变Au富集",
            "li_weathering": "风化淋滤Li流失"
        }
        st.session_state.selected_scene_key = st.selectbox(
            "选择预设场景",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options[x],
            index=list(scene_options.keys()).index(st.session_state.selected_scene_key)
        )
        selected_scene_key = st.session_state.selected_scene_key

        # 2. 加载场景（容错处理）
        if st.button("加载场景", type="primary"):
            try:
                scene_data = st.session_state.scene_manager.get_scene(selected_scene_key)
                if not scene_data:
                    st.error("场景加载失败，请重试！")
                else:
                    st.session_state.current_scene = scene_data
                    # 重置模拟对象
                    sim = st.session_state.sim
                    sim.reset_concentration()
                    initial_c = scene_data.get("initial_concentration", 0.01)
                    sim.concentration = np.full(sim.domain_size, initial_c)
                    # 中心点高浓度
                    center_x, center_y = sim.domain_size[0] // 2, sim.domain_size[1] // 2
                    sim.concentration[center_x - 5:center_x + 5, center_y - 5:center_y + 5] = initial_c * 10
                    sim.dt = scene_data.get("dt", 1.0)
                    st.session_state.sim_results = {}
                    st.success(f"成功加载：{scene_data.get('name', '未知场景')}")
            except Exception as e:
                st.error(f"加载场景出错：{str(e)}")

        st.divider()

        # 3. 参数调整（核心：彻底移除所有直接键读取，全部用get+默认值）
        current_scene = st.session_state.current_scene
        if current_scene:
            st.subheader("⚙️ 参数调整")

            # 温度（通用参数，容错）
            temp_range = current_scene.get("temperature_range", (0, 1000))
            default_temp = 300 if selected_scene_key == "au_hydrothermal" else 25
            temperature = st.slider(
                "温度 (℃)",
                min_value=temp_range[0],
                max_value=temp_range[1],
                value=default_temp,
                step=10
            )
            
            # PH值（Li场景0-12，容错）
            ph_range = current_scene.get("ph_range", (0.0, 12.0))
            default_ph = 5.0 if selected_scene_key == "au_hydrothermal" else 7.0
            ph = st.slider(
                "pH值",
                min_value=ph_range[0],
                max_value=ph_range[1],
                value=default_ph,
                step=0.1
            )

            # 场景专属参数
            additional_params = {}
            if selected_scene_key == "au_hydrothermal":
                # Au场景参数（全容错）
                pressure_range = current_scene.get("pressure_range", (10, 1000))
                eh_range = current_scene.get("eh_range", (-200, 400))
                sulfur_range = current_scene.get("sulfur_content_range", (0.01, 1.0))
                chlorine_range = current_scene.get("chlorine_content_range", (0.1, 10.0))
                
                pressure = st.slider("压力 (MPa)", pressure_range[0], pressure_range[1], 200, 10)
                eh = st.slider("氧化还原电位 (mV)", eh_range[0], eh_range[1], 100)
                sulfur_content = st.slider("硫含量 (wt%)", sulfur_range[0], sulfur_range[1], 0.5, 0.01)
                chlorine_content = st.slider("氯含量 (wt%)", chlorine_range[0], chlorine_range[1], 5.0, 0.1)
                
                additional_params = {
                    "pressure": pressure,
                    "eh": eh,
                    "sulfur_content": sulfur_content,
                    "chlorine_content": chlorine_content
                }
            elif selected_scene_key == "li_weathering":
                # 核心修复：彻底移除["water_mobility_range"]，全部用get+默认值
                mobility_range = current_scene.get("water_mobility_range", (0.1, 10.0))  # 永远不会KeyError
                water_mobility = st.slider(
                    "水的流动性（降水和水流）",
                    min_value=mobility_range[0],
                    max_value=mobility_range[1],
                    value=5.0,
                    step=0.1,
                    help="数值越大，Li元素随水流流失速度越快"
                )
                additional_params = {"water_mobility": water_mobility}

            # 模拟时间步长（容错）
            default_steps = 5000 if selected_scene_key == "au_hydrothermal" else 10000
            time_steps = st.slider(
                "模拟时间步长",
                min_value=100,
                max_value=20000,
                value=default_steps,
                step=100
            )

            # 保存参数
            st.session_state.params = {
                "temperature": temperature,
                "ph": ph,
                "time_steps": time_steps,** additional_params
            }

            # 4. 运行模拟（容错 + 确保Li场景结果完整赋值）
            if st.button("▶️ 运行模拟"):
                try:
                    with st.spinner("正在执行数值模拟..."):
                        sim = st.session_state.sim
                        scene = st.session_state.current_scene
                        params = st.session_state.params

                        # Li场景设置水流动性
                        if selected_scene_key == "li_weathering" and "water_mobility" in params:
                            sim.set_water_mobility(params["water_mobility"])

                        # 初始化模拟变量
                        time_points = []
                        avg_concentrations = []
                        solver_type = scene.get("solver_type", "explicit")
                        solver = sim.explicit_solver if solver_type == "explicit" else sim.implicit_solver
                        diffusion_coeff = scene.get("diffusion_coeff", 1e-6)
                        reaction_rate = scene.get("reaction_rate", 1e-4)

                        # 执行模拟（Li场景适配步长，避免无数据）
                        progress_bar = st.progress(0)
                        steps = int(params.get("time_steps", 5000))
                        # 确保至少记录10个数据点，避免时间曲线无数据
                        record_interval = max(1, steps // 100) if steps > 100 else 1
                        for step in range(steps):
                            solver(diffusion_coeff, reaction_rate)
                            if step % record_interval == 0:
                                time_points.append(sim.time)
                                avg_concentrations.append(np.mean(sim.concentration))
                            progress_bar.progress((step + 1) / steps)
                        progress_bar.empty()

                        # 生成结果（强制确保Li场景结果字段完整）
                        vis = ResultVisualization(sim)
                        initial_c = scene.get("initial_concentration", 0.01)
                        enrichment_factor = vis.calculate_enrichment_factor(initial_c)

                        # 保存结果（补充所有必要字段，避免展示时缺失）
                        st.session_state.sim_results = {
                            "enrichment_factor": enrichment_factor,
                            "simulation_time": sim.time,
                            "time_points": time_points if time_points else [0.0],  # 兜底空列表
                            "avg_concentrations": avg_concentrations if avg_concentrations else [initial_c],  # 兜底初始浓度
                            "water_mobility": params.get("water_mobility", 1.0),
                            "max_concentration": np.max(sim.concentration),
                            "min_concentration": np.min(sim.concentration)
                        }

                        st.success("模拟完成！结果已展示在主界面")
                except Exception as e:
                    st.error(f"模拟出错：{str(e)}")
                    # 模拟失败时也赋值基础结果，避免展示板块完全空白
                    st.session_state.sim_results = {
                        "enrichment_factor": 0.0,
                        "simulation_time": 0.0,
                        "time_points": [0.0],
                        "avg_concentrations": [0.0],
                        "water_mobility": params.get("water_mobility", 1.0),
                        "max_concentration": 0.0,
                        "min_concentration": 0.0
                    }

    # 右侧：结果展示板块（核心优化，确保Li场景正常显示）
    st.header("📊 模拟结果展示")

    # 优化判空逻辑：只要加载了场景就显示基础框架，模拟后显示完整结果
    if not st.session_state.current_scene:
        st.info("请先在左侧加载预设场景并运行模拟")
    else:
        sim_results = st.session_state.sim_results
        # 即使无模拟结果，也显示基础信息，避免空白
        if not sim_results:
            st.info(f"已加载【{st.session_state.current_scene.get('name', '未知场景')}】场景，请点击左侧「运行模拟」按钮生成结果")
        else:
            # 核心指标（适配Li场景的流失系数展示）
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_label = "流失系数" if selected_scene_key == "li_weathering" else "富集系数"
                st.metric(metric_label, f"{sim_results.get('enrichment_factor', 0):.2f}")
            with col2:
                st.metric("总模拟时间", f"{sim_results.get('simulation_time', 0):.0f}")
            with col3:
                max_c = sim_results.get("max_concentration", 0.0)
                st.metric("最高浓度", f"{max_c:.4f} ppm")
            with col4:
                # 修复1：场景名称从当前加载的场景获取，而非历史模拟结果
                st.metric("场景名称", st.session_state.current_scene.get('name', '未知场景'))

            # Li场景额外显示水流动性（强制显示，避免缺失）
            if selected_scene_key == "li_weathering":
                st.metric("水的流动性", f"{sim_results.get('water_mobility', 1.0):.1f}")

            st.divider()

            # 图表展示（容错 + 兜底数据，避免Li场景图表报错）
            try:
                vis = ResultVisualization(st.session_state.sim)
                tab1, tab2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
                with tab1:
                    contour_fig = vis.plot_contour()
                    st.pyplot(contour_fig)
                with tab2:
                    # 兜底数据：避免空列表导致图表报错
                    time_points = sim_results.get('time_points', [0.0])
                    avg_concs = sim_results.get('avg_concentrations', [0.0])
                    time_fig = vis.plot_time_series(time_points, avg_concs)
                    st.pyplot(time_fig)
            except Exception as e:
                st.error(f"图表生成出错：{str(e)}")
                # 图表生成失败时显示基础提示
                st.info("图表加载失败，核心模拟数据如下：")
                st.write(f"- 平均浓度：{np.mean(sim_results.get('avg_concentrations', [0.0])):.4f} ppm")
                st.write(f"- 模拟总时长：{sim_results.get('simulation_time', 0):.0f}")

            st.divider()

            # 数据导出（彻底修复Invalid binary data format错误 + 修复文件名）
            st.subheader("💾 数据导出")
            col_excel, col_vtk = st.columns(2)
            
            with col_excel:
                try:
                    vis = ResultVisualization(st.session_state.sim)
                    excel_bytes = vis.export_excel()  # 返回纯bytes
                    if excel_bytes:
                        # 修复2：导出文件名从当前加载的场景获取
                        scene_name = st.session_state.current_scene.get('name', '模拟结果').replace(" ", "_")
                        st.download_button(
                            label="导出Excel数据",
                            data=excel_bytes,  # 直接传纯字节数据
                            file_name=f"{scene_name}_浓度数据.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_btn"
                        )
                    else:
                        st.warning("Excel数据生成失败")
                except Exception as e:
                    st.error(f"Excel导出出错：{str(e)}")
            
            with col_vtk:
                try:
                    vis = ResultVisualization(st.session_state.sim)
                    vtk_str = vis.export_vtk()  # 返回纯字符串
                    if vtk_str:
                        # 修复3：导出文件名从当前加载的场景获取
                        scene_name = st.session_state.current_scene.get('name', '模拟结果').replace(" ", "_")
                        st.download_button(
                            label="导出VTK数据",
                            data=vtk_str,  # 直接传纯字符串
                            file_name=f"{scene_name}_浓度数据.vtk",
                            mime="text/plain",
                            key="vtk_btn"
                        )
                    else:
                        st.warning("VTK数据生成失败")
                except Exception as e:
                    st.error(f"VTK导出出错：{str(e)}")

# ===================== 运行程序（全局容错） =====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"程序运行出错：{str(e)}")
        # 重置会话状态
        st.session_state.clear()
        st.info("已重置会话状态，请刷新页面重新运行")
