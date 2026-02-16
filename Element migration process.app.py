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
def setup_chinese_font():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    font_paths = {
        'Windows': ['SimHei', 'Microsoft YaHei', 'FangSong'],
        'Linux': ['WenQuanYi Micro Hei', 'DejaVu Sans'],
        'Darwin': ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
    }
    system = platform.system()
    candidate_fonts = font_paths.get(system, ['DejaVu Sans'])
    available_fonts = [f for f in candidate_fonts if f in plt.rcParams['font.sans-serif']]
    plt.rcParams["font.family"] = available_fonts if available_fonts else ['DejaVu Sans']
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

setup_chinese_font()

st.set_page_config(
    page_title="地球化学元素迁移虚拟仿真平台",
    page_icon="🌍",
    layout="wide"
)

# ===================== 1. 数值模拟核心模块 =====================
class NumericalSimulation:
    def __init__(self, domain_size: Tuple[int, int] = (50, 50), dx: float = 1.0, dy: float = 1.0, dt: float = 1.0):
        self.domain_size = domain_size
        self.dx, self.dy = dx, dy
        self.dt = dt
        self.concentration = np.zeros(domain_size)
        self.time = 0.0
        self.saturation_concentration = 1.0
        self.water_mobility = 1.0

    def central_difference_x(self, field: np.ndarray) -> np.ndarray:
        return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * self.dx)

    def central_difference_y(self, field: np.ndarray) -> np.ndarray:
        return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * self.dy)

    def forward_difference_time(self, field: np.ndarray, rate: np.ndarray) -> np.ndarray:
        return field + rate * self.dt

    def explicit_solver(self, diffusion_coeff: float, reaction_rate: float) -> np.ndarray:
        laplacian = (
                (np.roll(self.concentration, -1, axis=1) + np.roll(self.concentration, 1, axis=1) - 2 * self.concentration) / self.dx**2 +
                (np.roll(self.concentration, -1, axis=0) + np.roll(self.concentration, 1, axis=0) - 2 * self.concentration) / self.dy**2
        )
        diffusion_term = diffusion_coeff * laplacian
        reaction_term = -reaction_rate * self.concentration
        self.concentration = self.forward_difference_time(self.concentration, diffusion_term + reaction_term)
        self.time += self.dt
        self.concentration = np.clip(self.concentration, 0, None)
        return self.concentration

    def implicit_solver(self, diffusion_coeff: float, reaction_rate: float, max_iter: int = 10) -> np.ndarray:
        new_concentration = self.concentration.copy()
        for _ in range(max_iter):
            for i in range(1, self.domain_size[0] - 1):
                for j in range(1, self.domain_size[1] - 1):
                    mobility_factor = self.water_mobility * 1e-2
                    denominator = 1 + self.dt * (2 * diffusion_coeff * (1/self.dx**2 + 1/self.dy**2) + reaction_rate)
                    if denominator < 1e-10:
                        denominator = 1e-10
                    new_concentration[i, j] = (
                        self.concentration[i, j] + self.dt * diffusion_coeff * (
                            (self.concentration[i+1,j] + self.concentration[i-1,j])/self.dx**2 +
                            (self.concentration[i,j+1] + self.concentration[i,j-1])/self.dy**2
                        ) - mobility_factor * self.concentration[i,j]
                    ) / denominator
        self.concentration = np.clip(new_concentration, 0, np.max(new_concentration))
        self.time += self.dt
        return self.concentration

    def set_water_mobility(self, mobility: float):
        self.water_mobility = mobility

    def reset_concentration(self):
        self.concentration = np.zeros(self.domain_size)
        self.time = 0.0
        self.water_mobility = 1.0

# ===================== 2. 场景预设模块 =====================
class SceneManager:
    def __init__(self):
        self.scenes: Dict[str, Dict] = {
            "au_hydrothermal": {
                "name": "热液蚀变Au富集",
                "initial_concentration": 0.01,
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
                "initial_concentration": 50,
                "ph_range": (0.0, 12.0),
                "temperature_range": (0, 1000),
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "water_mobility_range": (0.1, 10.0),
                "time_range": (1000, 100000),
                "dt": 100.0,
                "diffusion_coeff": 1e-7,
                "reaction_rate": 1e-5,
                "solver_type": "implicit"
            }
        }

    def get_scene(self, scene_name: str) -> Dict:
        default_scene = {
            "name": "未知场景",
            "initial_concentration": 0.01,
            "temperature_range": (0, 1000),
            "ph_range": (0.0, 12.0),
            "pressure_range": (10, 1000),
            "eh_range": (-200, 400),
            "sulfur_content_range": (0.01, 1.0),
            "chlorine_content_range": (0.1, 10.0),
            "water_mobility_range": (0.1, 10.0),
            "time_range": (100, 20000),
            "dt": 1.0,
            "diffusion_coeff": 1e-6,
            "reaction_rate": 1e-4,
            "solver_type": "explicit"
        }
        scene = self.scenes.get(scene_name, {})
        for key in default_scene:
            scene.setdefault(key, default_scene[key])
        return scene

# ===================== 3. 结果可视化模块 =====================
class ResultVisualization:
    def __init__(self, simulation: NumericalSimulation):
        self.simulation = simulation
        setup_chinese_font()

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor="white")
        fig.suptitle("Concentration Contour Map", fontsize=14, fontweight='bold')

        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        if max_c - min_c < 1e-6:
            max_c = min_c + 10.0
        levels = np.linspace(min_c, max_c, 20)

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
        cbar = fig.colorbar(contour, ax=ax, label='Concentration (ppm)', shrink=0.8)
        cbar.ax.set_ylabel('Concentration (ppm)', fontsize=10)
        ax.set_xlabel('Spatial Coordinate X', fontsize=12)
        ax.set_ylabel('Spatial Coordinate Y', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        return fig

    def plot_time_series(self, time_points: List[float], concentrations: List[float], title: str = "浓度-时间曲线") -> plt.Figure:
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
        
        if not time_points or not concentrations:
            time_points = [0, 1000, 2000]
            concentrations = [50, 40, 30]
            st.warning("时间/浓度数据为空，使用默认数据绘图")
        
        ax.plot(time_points, concentrations, 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Average Concentration (ppm)', fontsize=12)
        ax.set_title("Concentration-Time Curve", fontsize=14, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        return fig

    def calculate_enrichment_factor(self, initial_concentration: float) -> float:
        max_concentration = np.max(self.simulation.concentration)
        if initial_concentration <= 0:
            return 0.0
        factor = max_concentration / initial_concentration
        if "li_weathering" in st.session_state.get("current_scene", {}).get("name", ""):
            return 1.0 / factor if factor > 0 else 100.0
        return factor

    def export_excel(self) -> bytes:
        try:
            import openpyxl
        except ImportError:
            st.error("缺少Excel依赖：pip install openpyxl")
            return b""
        
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
        
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='浓度数据', index=False)
            output.seek(0)
            excel_data = output.getvalue()
            output.close()
            return excel_data
        except Exception as e:
            st.error(f"Excel导出失败：{str(e)}")
            return b""

    def export_vtk(self) -> str:
        nx, ny = self.simulation.domain_size
        n_points = nx * ny
        
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
        for j in range(ny):
            for i in range(nx):
                vtk_content += f"{self.simulation.concentration[i, j]:.6f}\n"
        
        return vtk_content

# ===================== 4. 教学管理模块 =====================
class TeachingManagement:
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

# ===================== 5. 会话状态初始化 =====================
def init_session_state():
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation(domain_size=(50, 50), dx=1.0, dy=1.0, dt=1.0)
    if "scene_manager" not in st.session_state:
        st.session_state.scene_manager = SceneManager()
    if "teaching_manager" not in st.session_state:
        st.session_state.teaching_manager = TeachingManagement()
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
        st.session_state.current_scene = st.session_state.scene_manager.get_scene("au_hydrothermal")
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = {}
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "selected_scene_key" not in st.session_state:
        st.session_state.selected_scene_key = "au_hydrothermal"

# ===================== 6. 主界面逻辑（最终修复） =====================
def main():
    init_session_state()

    # 调试面板（可选显示）
    with st.expander("🔍 调试信息（点击展开）", expanded=False):
        st.write("**会话状态检查**")
        st.write(f"- 当前场景：{st.session_state.get('selected_scene_key', '未选择')}")
        st.write(f"- sim_results是否为空：{len(st.session_state.get('sim_results', {})) == 0}")
        st.write(f"- sim_results内容：{st.session_state.get('sim_results', {})}")
        st.write(f"- 模拟对象dt：{st.session_state.sim.dt if 'sim' in st.session_state else '未初始化'}")
        if st.session_state.get('current_scene') is not None:
            st.write(f"- 初始浓度：{st.session_state.current_scene.get('initial_concentration', '未知')}")
        else:
            st.write("- 初始浓度：未加载场景")

    st.title("🌍 地球化学元素迁移虚拟仿真平台")
    st.divider()

    with st.sidebar:
        st.header("🔧 实验配置")

        # 1. 场景选择（绑定会话状态）
        scene_options = {
            "au_hydrothermal": "热液蚀变Au富集",
            "li_weathering": "风化淋滤Li流失"
        }
        new_selected_scene_key = st.selectbox(
            "选择预设场景",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options[x],
            index=list(scene_options.keys()).index(st.session_state.selected_scene_key)
        )

        # 核心修复：场景切换时强制重置current_scene和sim_results
        if new_selected_scene_key != st.session_state.selected_scene_key:
            st.session_state.selected_scene_key = new_selected_scene_key
            # 立即加载新场景，确保current_scene始终与selected_scene_key一致
            st.session_state.current_scene = st.session_state.scene_manager.get_scene(new_selected_scene_key)
            st.session_state.sim_results = {}
            st.rerun()

        selected_scene_key = st.session_state.selected_scene_key

        # 2. 加载场景（容错处理）
        if st.button("加载场景", type="primary"):
            try:
                # 直接根据selected_scene_key加载场景，确保current_scene与选择一致
                scene_data = st.session_state.scene_manager.get_scene(selected_scene_key)
                st.session_state.current_scene = scene_data
                
                # 重置并初始化模拟对象
                sim = st.session_state.sim
                sim.reset_concentration()
                initial_c = scene_data["initial_concentration"]
                sim.concentration = np.full(sim.domain_size, initial_c)
                center_x, center_y = sim.domain_size[0] // 2, sim.domain_size[1] // 2
                sim.concentration[center_x-5:center_x+5, center_y-5:center_y+5] = initial_c * 10
                sim.dt = scene_data["dt"]
                
                st.session_state.sim_results = {}
                st.success(f"成功加载：{scene_data['name']}")
                st.debug(f"加载场景后初始浓度：min={np.min(sim.concentration):.2f}, max={np.max(sim.concentration):.2f}")
            except Exception as e:
                st.error(f"加载场景出错：{str(e)}")
                st.exception(e)

        st.divider()

        # 3. 参数调整
        current_scene = st.session_state.current_scene
        if current_scene:
            st.subheader("⚙️ 参数调整")

            # 温度
            temp_range = current_scene["temperature_range"]
            default_temp = 300 if selected_scene_key == "au_hydrothermal" else 25
            temperature = st.slider("温度 (℃)", temp_range[0], temp_range[1], default_temp, 10)
            
            # PH值
            ph_range = current_scene["ph_range"]
            default_ph = 5.0 if selected_scene_key == "au_hydrothermal" else 7.0
            ph = st.slider("pH值", ph_range[0], ph_range[1], default_ph, 0.1)

            # 场景专属参数
            additional_params = {}
            if selected_scene_key == "au_hydrothermal":
                pressure_range = current_scene["pressure_range"]
                eh_range = current_scene["eh_range"]
                sulfur_range = current_scene["sulfur_content_range"]
                chlorine_range = current_scene["chlorine_content_range"]
                
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
                mobility_range = current_scene["water_mobility_range"]
                water_mobility = st.slider(
                    "水的流动性（降水和水流）",
                    min_value=mobility_range[0],
                    max_value=mobility_range[1],
                    value=5.0,
                    step=0.1,
                    help="数值越大，Li元素随水流流失速度越快"
                )
                additional_params = {"water_mobility": water_mobility}

            # 模拟时间步长
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
                "time_steps": time_steps,
                **additional_params
            }

            # 4. 运行模拟
            if st.button("▶️ 运行模拟"):
                try:
                    with st.spinner("正在执行数值模拟..."):
                        sim = st.session_state.sim
                        scene = st.session_state.current_scene
                        params = st.session_state.params

                        if selected_scene_key == "li_weathering" and "water_mobility" in params:
                            sim.set_water_mobility(params["water_mobility"])

                        time_points = []
                        avg_concentrations = []
                        solver_type = scene["solver_type"]
                        solver = sim.explicit_solver if solver_type == "explicit" else sim.implicit_solver
                        diffusion_coeff = scene["diffusion_coeff"]
                        reaction_rate = scene["reaction_rate"]

                        progress_bar = st.progress(0)
                        steps = int(params["time_steps"])
                        record_interval = max(1, steps // 100) if steps > 100 else 1
                        for step in range(steps):
                            solver(diffusion_coeff, reaction_rate)
                            if step % record_interval == 0 or step == steps-1:
                                time_points.append(sim.time)
                                avg_concentrations.append(np.mean(sim.concentration))
                            progress_bar.progress((step + 1) / steps)
                        progress_bar.empty()

                        vis = ResultVisualization(sim)
                        initial_c = scene["initial_concentration"]
                        enrichment_factor = vis.calculate_enrichment_factor(initial_c)

                        st.session_state.sim_results = {
                            "enrichment_factor": enrichment_factor,
                            "simulation_time": sim.time,
                            "time_points": time_points if time_points else [0.0, sim.time],
                            "avg_concentrations": avg_concentrations if avg_concentrations else [initial_c, np.mean(sim.concentration)],
                            "water_mobility": params.get("water_mobility", 1.0),
                            "max_concentration": np.max(sim.concentration),
                            "min_concentration": np.min(sim.concentration)
                        }

                        st.success("模拟完成！结果已展示在主界面")
                        st.debug(f"模拟结果：{st.session_state.sim_results}")
                except Exception as e:
                    st.error(f"模拟出错：{str(e)}")
                    st.exception(e)
                    st.session_state.sim_results = {
                        "enrichment_factor": 0.0,
                        "simulation_time": 0.0,
                        "time_points": [0.0, 1000],
                        "avg_concentrations": [50.0, 40.0],
                        "water_mobility": params.get("water_mobility", 1.0),
                        "max_concentration": 0.0,
                        "min_concentration": 0.0
                    }

    # 右侧：结果展示板块
    st.header("📊 模拟结果展示")

    if not st.session_state.current_scene:
        st.info("请先在左侧加载预设场景并运行模拟")
    else:
        sim_results = st.session_state.sim_results
        if not sim_results:
            st.info(f"已加载【{st.session_state.current_scene['name']}】场景，请点击左侧「运行模拟」按钮生成结果")
        else:
            # 核心指标（场景名称直接从current_scene获取）
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_label = "流失系数" if selected_scene_key == "li_weathering" else "富集系数"
                st.metric(metric_label, f"{sim_results.get('enrichment_factor', 0.0):.2f}")
            with col2:
                st.metric("总模拟时间", f"{sim_results.get('simulation_time', 0.0):.0f}")
            with col3:
                st.metric("最高浓度", f"{sim_results.get('max_concentration', 0.0):.4f} ppm")
            with col4:
                # 最终修复：场景名称直接从current_scene获取，确保与当前选择一致
                st.metric("场景名称", st.session_state.current_scene["name"])

            if selected_scene_key == "li_weathering":
                st.metric("水的流动性", f"{sim_results.get('water_mobility', 1.0):.1f}")

            st.divider()

            # 图表展示
            try:
                sim = st.session_state.sim
                vis = ResultVisualization(sim)
                tab1, tab2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
                with tab1:
                    contour_fig = vis.plot_contour()
                    st.pyplot(contour_fig)
                with tab2:
                    time_fig = vis.plot_time_series(
                        sim_results.get('time_points', [0, 1000]),
                        sim_results.get('avg_concentrations', [50, 40])
                    )
                    st.pyplot(time_fig)
            except Exception as e:
                st.error(f"图表生成出错：{str(e)}")
                st.exception(e)
                st.info(f"核心数据：平均浓度 {np.mean(sim_results.get('avg_concentrations', [0])):.4f} ppm")

            st.divider()

            # 数据导出（文件名直接从current_scene获取）
            st.subheader("💾 数据导出")
            col_excel, col_vtk = st.columns(2)
            
            with col_excel:
                try:
                    sim = st.session_state.sim
                    vis = ResultVisualization(sim)
                    excel_data = vis.export_excel()
                    if excel_data:
                        # 最终修复：导出文件名从current_scene获取
                        scene_name = st.session_state.current_scene["name"].replace(" ", "_")
                        st.download_button(
                            label="导出Excel数据",
                            data=excel_data,
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
                    sim = st.session_state.sim
                    vis = ResultVisualization(sim)
                    vtk_data = vis.export_vtk()
                    if vtk_data:
                        # 最终修复：导出文件名从current_scene获取
                        scene_name = st.session_state.current_scene["name"].replace(" ", "_")
                        st.download_button(
                            label="导出VTK数据",
                            data=vtk_data,
                            file_name=f"{scene_name}_浓度数据.vtk",
                            mime="text/plain",
                            key="vtk_btn"
                        )
                    else:
                        st.warning("VTK数据生成失败")
                except Exception as e:
                    st.error(f"VTK导出出错：{str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"程序运行出错：{str(e)}")
        st.exception(e)
        st.session_state.clear()
        st.info("已重置会话状态，请刷新页面重新运行")
