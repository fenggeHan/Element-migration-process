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
    import platform
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
        """隐式有限差分求解（Jacobi迭代）"""
        new_concentration = self.concentration.copy()
        for _ in range(max_iter):
            for i in range(1, self.domain_size[0] - 1):
                for j in range(1, self.domain_size[1] - 1):
                    # 隐式格式离散
                    new_concentration[i, j] = (
                                                      self.concentration[i, j] + self.dt * diffusion_coeff * (
                                                      (self.concentration[i + 1, j] + self.concentration[
                                                          i - 1, j]) / self.dx ** 2 +
                                                      (self.concentration[i, j + 1] + self.concentration[
                                                          i, j - 1]) / self.dy ** 2
                                              )
                                              ) / (1 + self.dt * (
                            2 * diffusion_coeff * (1 / self.dx ** 2 + 1 / self.dy ** 2) + reaction_rate))
        self.concentration = new_concentration
        self.concentration = np.clip(self.concentration, 0, None)
        self.time += self.dt
        return self.concentration

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

# ===================== 2. 场景预设与自定义模块 =====================
class SceneManager:
    """管理内置场景与自定义场景"""

    def __init__(self):
        self.scenes: Dict[str, Dict] = {
            "au_hydrothermal": {
                "name": "热液蚀变Au富集",
                "initial_concentration": 0.01,  # ppm
                "temperature_range": (0, 1000),
                "ph_range": (2.0, 8.0),
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
                "ph_range": (3.0, 5.0),
                "temperature_range": (0, 1000),  # 补充缺失的参数
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "time_range": (1000, 100000),
                "dt": 100.0,
                "diffusion_coeff": 1e-7,
                "reaction_rate": 1e-5,
                "solver_type": "implicit"
            }
        }

    def get_scene(self, scene_name: str) -> Dict:
        """获取场景参数"""
        return self.scenes.get(scene_name, {})

    def create_custom_scene(self, name: str, params: Dict) -> Dict:
        """创建自定义场景"""
        self.scenes[name] = params
        return self.scenes[name]

# ===================== 3. 结果可视化与分析模块 =====================
class ResultVisualization:
    """结果可视化与分析工具（适配Streamlit）"""

    def __init__(self, simulation: NumericalSimulation):
        self.simulation = simulation
        # 每次初始化都重新配置字体
        setup_chinese_font()

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        """重构等值线图绘制逻辑，确保显示正常"""
        # 强制重新配置字体
        setup_chinese_font()
        
        # 创建全新的figure对象，避免缓存冲突
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor="white")
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 生成浓度等值线
        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        
        # 确保有足够的梯度
        if max_c - min_c < 1e-6:
            levels = np.linspace(min_c, min_c + 0.02, 20)
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
        
        # 添加等值线轮廓
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
        
        # 确保布局紧凑
        plt.tight_layout()
        
        return fig

    def plot_time_series(self, time_points: List[float], concentrations: List[float],
                         title: str = "Concentration-Time Curve") -> plt.Figure:
        """绘制浓度随时间变化曲线"""
        # 强制重新配置字体
        setup_chinese_font()
        
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
        
        ax.plot(time_points, concentrations, 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Average Concentration (ppm)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        return fig

    def calculate_enrichment_factor(self, initial_concentration: float) -> float:
        """计算元素富集系数"""
        max_concentration = np.max(self.simulation.concentration)
        return max_concentration / initial_concentration if initial_concentration > 0 else 0.0

    def export_csv(self) -> StringIO:
        """导出浓度场数据为CSV（修复数据流处理）"""
        output = StringIO()
        # 使用pandas更稳定地生成CSV
        df = pd.DataFrame()
        x_coords, y_coords, concs = [], [], []
        
        for i in range(self.simulation.domain_size[0]):
            for j in range(self.simulation.domain_size[1]):
                x_coords.append(i)
                y_coords.append(j)
                concs.append(self.simulation.concentration[i, j])
        
        df['X_Coordinate'] = x_coords
        df['Y_Coordinate'] = y_coords
        df['Concentration_(ppm)'] = concs
        
        # 写入CSV
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return output

    def export_vtk(self) -> StringIO:
        """导出浓度场数据为VTK格式（修复格式错误）"""
        output = StringIO()
        nx, ny = self.simulation.domain_size
        n_points = nx * ny
        
        # 标准VTK结构化点格式
        vtk_header = f"""# vtk DataFile Version 3.0
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
        output.write(vtk_header)
        
        # 按VTK要求的顺序写入数据（先Y后X）
        for j in range(ny):
            for i in range(nx):
                output.write(f"{self.simulation.concentration[i, j]:.6f}\n")
        
        output.seek(0)
        return output

# ===================== 4. 教学管理模块 =====================
class TeachingManagement:
    """教学任务管理与数据统计"""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.student_data: Dict[str, List[str]] = {}

    def create_task(self, task_id: str, scene_name: str, param_ranges: Dict, deadline: str) -> None:
        """创建教学实验任务"""
        self.tasks[task_id] = {
            "scene_name": scene_name,
            "param_ranges": param_ranges,
            "deadline": deadline,
            "submissions": {}
        }

    def submit_experiment(self, task_id: str, student_id: str, params: Dict, results: Dict) -> None:
        """学生提交实验报告"""
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
        """自动批改实验报告"""
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
        """导出任务统计数据"""
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

# ===================== 5. Streamlit 交互界面主逻辑 =====================
def main():
    # 初始化会话状态
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
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = {}
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = {}
    if "params" not in st.session_state:
        st.session_state.params = {}

    # ===== 页面标题与布局 =====
    st.title("🌍 地球化学元素迁移虚拟仿真平台")
    st.divider()

    # ===== 左侧：场景选择与参数配置 =====
    with st.sidebar:
        st.header("🔧 实验配置")

        # 1. 场景选择
        scene_options = {
            "au_hydrothermal": "热液蚀变Au富集",
            "li_weathering": "风化淋滤Li流失"
        }
        selected_scene_key = st.selectbox(
            "选择预设场景",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options[x]
        )

        # 加载选中场景
        if st.button("加载场景", type="primary"):
            st.session_state.current_scene = st.session_state.scene_manager.get_scene(selected_scene_key)
            # 重置并初始化浓度场
            sim = st.session_state.sim
            sim.reset_concentration()  # 重置
            initial_c = st.session_state.current_scene["initial_concentration"]
            sim.concentration = np.full(sim.domain_size, initial_c)
            # 中心点设置高浓度
            center_x, center_y = sim.domain_size[0] // 2, sim.domain_size[1] // 2
            sim.concentration[center_x - 5:center_x + 5, center_y - 5:center_y + 5] = initial_c * 10
            sim.dt = st.session_state.current_scene["dt"]
            # 清空之前的模拟结果
            st.session_state.sim_results = {}
            st.success(f"成功加载场景：{st.session_state.current_scene['name']}")

        st.divider()

        # 2. 参数调整
        if st.session_state.current_scene:
            st.subheader("⚙️ 参数调整")

            # 温度
            temperature = st.slider(
                "温度 (℃)",
                min_value=st.session_state.current_scene["temperature_range"][0],
                max_value=st.session_state.current_scene["temperature_range"][1],
                value=300,
                step=10
            )
            
            # pH值
            ph = st.slider(
                "pH值",
                min_value=st.session_state.current_scene["ph_range"][0],
                max_value=st.session_state.current_scene["ph_range"][1],
                value=5.0,
                step=0.1
            )

            # 场景专属参数
            additional_params = {}
            if selected_scene_key == "au_hydrothermal":
                pressure = st.slider(
                    "压力 (MPa)",
                    min_value=st.session_state.current_scene["pressure_range"][0],
                    max_value=st.session_state.current_scene["pressure_range"][1],
                    value=200,
                    step=10
                )
                eh = st.slider(
                    "氧化还原电位 (mV)",
                    min_value=st.session_state.current_scene["eh_range"][0],
                    max_value=st.session_state.current_scene["eh_range"][1],
                    value=100
                )
                sulfur_content = st.slider(
                    "硫含量 (wt%)",
                    min_value=st.session_state.current_scene["sulfur_content_range"][0],
                    max_value=st.session_state.current_scene["sulfur_content_range"][1],
                    value=0.5,
                    step=0.01
                )
                chlorine_content = st.slider(
                    "氯含量 (wt%)",
                    min_value=st.session_state.current_scene["chlorine_content_range"][0],
                    max_value=st.session_state.current_scene["chlorine_content_range"][1],
                    value=5.0,
                    step=0.1
                )
                additional_params = {
                    "pressure": pressure,
                    "eh": eh,
                    "sulfur_content": sulfur_content,
                    "chlorine_content": chlorine_content
                }

            # 模拟时间步长
            time_steps = st.slider(
                "模拟时间步长",
                min_value=100,
                max_value=20000,
                value=5000,
                step=100
            )

            # 保存参数
            st.session_state.params = {
                "temperature": temperature,
                "ph": ph,
                "time_steps": time_steps,
                **additional_params
            }

            # 3. 运行模拟
            if st.button("▶️ 运行模拟"):
                with st.spinner("正在执行数值模拟..."):
                    sim = st.session_state.sim
                    scene = st.session_state.current_scene
                    params = st.session_state.params

                    time_points = []
                    avg_concentrations = []
                    solver = sim.explicit_solver if scene["solver_type"] == "explicit" else sim.implicit_solver

                    # 执行模拟
                    progress_bar = st.progress(0)
                    for step in range(int(params["time_steps"])):
                        solver(scene["diffusion_coeff"], scene["reaction_rate"])
                        if step % 200 == 0:
                            time_points.append(sim.time)
                            avg_concentrations.append(np.mean(sim.concentration))
                        progress_bar.progress((step + 1) / int(params["time_steps"]))
                    progress_bar.empty()

                    # 生成可视化结果
                    vis = ResultVisualization(sim)
                    contour_fig = vis.plot_contour(title=f"{scene['name']} - 浓度等值线图")
                    time_fig = vis.plot_time_series(time_points, avg_concentrations,
                                                    title=f"{scene['name']} - 浓度-时间曲线")

                    # 计算核心指标
                    enrichment_factor = vis.calculate_enrichment_factor(scene["initial_concentration"])

                    # 保存结果（只存关键数据，不缓存大对象）
                    st.session_state.sim_results = {
                        "enrichment_factor": enrichment_factor,
                        "simulation_time": sim.time,
                        "time_points": time_points,
                        "avg_concentrations": avg_concentrations,
                        "scene_name": scene["name"]
                    }

                    st.success("模拟完成！结果已展示在主界面")

    # ===== 右侧：结果展示 =====
    st.header("📊 模拟结果展示")

    if not st.session_state.current_scene:
        st.info("请先在左侧加载预设场景并运行模拟")
    else:
        if st.session_state.sim_results:
            # 核心指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("富集系数", f"{st.session_state.sim_results['enrichment_factor']:.2f}")
            with col2:
                st.metric("总模拟时间", f"{st.session_state.sim_results['simulation_time']:.0f}")
            with col3:
                st.metric("最高浓度", f"{np.max(st.session_state.sim.concentration):.4f} ppm")
            with col4:
                st.metric("场景名称", st.session_state.sim_results['scene_name'])

            st.divider()

            # 动态生成可视化图表（避免缓存问题）
            vis = ResultVisualization(st.session_state.sim)
            tab1, tab2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
            with tab1:
                contour_fig = vis.plot_contour(title=f"{st.session_state.sim_results['scene_name']} - 浓度等值线图")
                st.pyplot(contour_fig)  # 移除clear_figure=True
            with tab2:
                time_fig = vis.plot_time_series(
                    st.session_state.sim_results['time_points'],
                    st.session_state.sim_results['avg_concentrations'],
                    title=f"{st.session_state.sim_results['scene_name']} - 浓度-时间曲线"
                )
                st.pyplot(time_fig)

            st.divider()

            # 数据导出（实时生成，不依赖缓存）
            st.subheader("💾 数据导出")
            col_csv, col_vtk = st.columns(2)
            
            with col_csv:
                # 实时生成CSV数据
                csv_data = vis.export_csv()
                st.download_button(
                    label="导出CSV数据",
                    data=csv_data,
                    file_name=f"{st.session_state.sim_results['scene_name']}_浓度数据.csv",
                    mime="text/csv; charset=utf-8"
                )
            
            with col_vtk:
                # 实时生成VTK数据
                vtk_data = vis.export_vtk()
                st.download_button(
                    label="导出VTK数据",
                    data=vtk_data,
                    file_name=f"{st.session_state.sim_results['scene_name']}_浓度数据.vtk",
                    mime="text/plain"
                )

    # ===== 教学管理模块 =====
    with st.expander("🎓 教学管理功能（教师端）", expanded=False):
        student_id = st.text_input("学生ID")
        task_id = st.text_input("任务ID", value="GEOCHEM_TASK_001")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("提交实验报告") and student_id and st.session_state.sim_results:
                st.session_state.teaching_manager.submit_experiment(
                    task_id=task_id,
                    student_id=student_id,
                    params=st.session_state.params,
                    results={
                        "enrichment_factor": st.session_state.sim_results["enrichment_factor"],
                        "simulation_time": st.session_state.sim_results["simulation_time"]
                    }
                )
                st.success(f"学生 {student_id} 已提交任务 {task_id} 的实验报告")

        with col2:
            if st.button("自动批改") and student_id:
                grade, comment = st.session_state.teaching_manager.auto_grade(task_id, student_id)
                st.write(f"**批改结果**：{grade}")
                st.write(f"**评语**：{comment}")

        with col3:
            if st.button("导出统计数据"):
                stats = st.session_state.teaching_manager.export_statistics(task_id)
                if stats:
                    st.write("### 任务统计数据")
                    st.json(stats)
                else:
                    st.warning("该任务无统计数据")

if __name__ == "__main__":
    main()

