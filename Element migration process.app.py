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
    """跨平台中文字体配置"""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    font_paths = {
        'Windows': ['SimHei', 'Microsoft YaHei'],
        'Linux': ['WenQuanYi Micro Hei', 'DejaVu Sans'],
        'Darwin': ['PingFang SC', 'Arial Unicode MS']
    }
    system = platform.system()
    candidate_fonts = font_paths.get(system, ['DejaVu Sans'])
    available_fonts = [f for f in candidate_fonts if f in plt.rcParams['font.sans-serif']]
    plt.rcParams["font.family"] = available_fonts if available_fonts else ['DejaVu Sans']
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150

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
        self.water_mobility = 1.0  # 水的流动性参数

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
                    new_concentration[i, j] = (
                        self.concentration[i, j] + self.dt * diffusion_coeff * (
                            (self.concentration[i+1, j] + self.concentration[i-1, j])/self.dx**2 +
                            (self.concentration[i, j+1] + self.concentration[i, j-1])/self.dy**2
                        ) - mobility_factor * self.concentration[i, j]
                    ) / (1 + self.dt * (2 * diffusion_coeff * (1/self.dx**2 + 1/self.dy**2) + reaction_rate))
        self.concentration = new_concentration
        self.concentration = np.clip(self.concentration, 0, None)
        self.time += self.dt
        return self.concentration

    def set_water_mobility(self, mobility: float):
        self.water_mobility = mobility

    def reset_concentration(self):
        self.concentration = np.zeros(self.domain_size)
        self.time = 0.0
        self.water_mobility = 1.0

# ===================== 2. 场景预设模块（确保参数存在） =====================
class SceneManager:
    def __init__(self):
        # 强制初始化所有场景参数，包含water_mobility_range
        self.scenes = {
            "au_hydrothermal": {
                "name": "热液蚀变Au富集",
                "initial_concentration": 0.01,
                "temperature_range": (0, 1000),
                "ph_range": (2.0, 8.0),
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "dt": 1.0,
                "diffusion_coeff": 1e-6,
                "reaction_rate": 1e-4,
                "solver_type": "explicit"
            },
            "li_weathering": {
                "name": "风化淋滤Li流失",
                "initial_concentration": 50,
                "temperature_range": (0, 1000),
                "ph_range": (0.0, 12.0),  # PH范围0-12
                "water_mobility_range": (0.1, 10.0),  # 强制存在该参数
                "dt": 100.0,
                "diffusion_coeff": 1e-7,
                "reaction_rate": 1e-5,
                "solver_type": "implicit"
            }
        }

    def get_scene(self, scene_name):
        # 安全获取场景，缺失则返回空字典
        scene = self.scenes.get(scene_name, {})
        # 为Li场景强制补充water_mobility_range（核心修复）
        if scene_name == "li_weathering":
            scene["water_mobility_range"] = scene.get("water_mobility_range", (0.1, 10.0))
            scene["ph_range"] = scene.get("ph_range", (0.0, 12.0))
        return scene

# ===================== 3. 可视化与导出模块（修复格式错误） =====================
class ResultVisualization:
    def __init__(self, simulation):
        self.simulation = simulation
        setup_chinese_font()

    def plot_contour(self):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        fig.suptitle("Concentration Contour Map", fontsize=14)
        
        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        levels = np.linspace(min_c, max_c if max_c - min_c > 1e-6 else min_c + 5, 20)
        
        contour = ax.contourf(self.simulation.concentration, levels=levels, cmap='viridis', alpha=0.8)
        ax.contour(self.simulation.concentration, levels=levels, colors='white', linewidths=0.5)
        fig.colorbar(contour, ax=ax, label='Concentration (ppm)')
        
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        plt.tight_layout()
        return fig

    def plot_time_series(self, time_points, concentrations):
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(time_points, concentrations, 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Concentration (ppm)')
        ax.set_title("Concentration-Time Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def calculate_enrichment_factor(self, initial_c):
        max_c = np.max(self.simulation.concentration)
        factor = max_c / initial_c if initial_c > 0 else 0
        return 1/factor if "li_weathering" in st.session_state.get("current_scene", {}).get("name", "") and factor > 0 else factor

    def export_excel(self):
        try:
            import openpyxl
        except ImportError:
            st.error("请安装：pip install openpyxl")
            return b""
        
        x, y, c = [], [], []
        nx, ny = self.simulation.domain_size
        for i in range(nx):
            for j in range(ny):
                x.append(i), y.append(j), c.append(float(self.simulation.concentration[i,j]))
        
        df = pd.DataFrame({'X坐标':x, 'Y坐标':y, '浓度(ppm)':c})
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return output.getvalue()  # 返回纯bytes

    def export_vtk(self):
        nx, ny = self.simulation.domain_size
        vtk = f"""# vtk DataFile Version 3.0
Geochemical Simulation
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS {ny} {nx} 1
ORIGIN 0 0 0
SPACING {self.simulation.dx} {self.simulation.dy} 1
POINT_DATA {nx*ny}
SCALARS concentration float 1
LOOKUP_TABLE default
"""
        for j in range(ny):
            for i in range(nx):
                vtk += f"{self.simulation.concentration[i,j]:.6f}\n"
        return vtk  # 返回纯字符串

# ===================== 4. 会话状态初始化（核心） =====================
def init_session():
    """强制初始化所有会话状态，避免未定义"""
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation()
    if "scene_manager" not in st.session_state:
        st.session_state.scene_manager = SceneManager()
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = {}
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = {}
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "selected_scene" not in st.session_state:
        st.session_state.selected_scene = "au_hydrothermal"

# ===================== 5. 主程序（移除所有直接键读取） =====================
def main():
    init_session()

    # 页面标题
    st.title("🌍 地球化学元素迁移虚拟仿真平台")
    st.divider()

    # 左侧配置栏
    with st.sidebar:
        st.header("🔧 实验配置")

        # 1. 场景选择
        scene_options = {"au_hydrothermal": "热液蚀变Au富集", "li_weathering": "风化淋滤Li流失"}
        st.session_state.selected_scene = st.selectbox(
            "选择场景",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options[x],
            index=list(scene_options.keys()).index(st.session_state.selected_scene)
        )
        selected_scene = st.session_state.selected_scene

        # 2. 加载场景
        if st.button("加载场景", type="primary"):
            try:
                scene_data = st.session_state.scene_manager.get_scene(selected_scene)
                st.session_state.current_scene = scene_data
                # 重置模拟
                sim = st.session_state.sim
                sim.reset_concentration()
                initial_c = scene_data.get("initial_concentration", 0.01)
                sim.concentration = np.full(sim.domain_size, initial_c)
                center = (sim.domain_size[0]//2, sim.domain_size[1]//2)
                sim.concentration[center[0]-5:center[0]+5, center[1]-5:center[1]+5] = initial_c * 10
                sim.dt = scene_data.get("dt", 1.0)
                st.session_state.sim_results = {}
                st.success(f"加载成功：{scene_data.get('name', '未知场景')}")
            except Exception as e:
                st.error(f"加载失败：{str(e)}")

        st.divider()

        # 3. 参数调整（全部使用get方法，无直接键读取）
        current_scene = st.session_state.current_scene
        if current_scene:
            st.subheader("⚙️ 参数调整")

            # 温度（通用）
            temp_range = current_scene.get("temperature_range", (0, 1000))
            temperature = st.slider("温度 (℃)", temp_range[0], temp_range[1], 300 if selected_scene == "au_hydrothermal" else 25, 10)

            # PH值（Li场景0-12）
            ph_range = current_scene.get("ph_range", (0.0, 12.0))
            ph = st.slider("pH值", ph_range[0], ph_range[1], 5.0 if selected_scene == "au_hydrothermal" else 7.0, 0.1)

            # 场景专属参数
            additional_params = {}
            if selected_scene == "au_hydrothermal":
                pressure_range = current_scene.get("pressure_range", (10, 1000))
                eh_range = current_scene.get("eh_range", (-200, 400))
                sulfur_range = current_scene.get("sulfur_content_range", (0.01, 1.0))
                chlorine_range = current_scene.get("chlorine_content_range", (0.1, 10.0))
                
                additional_params = {
                    "pressure": st.slider("压力 (MPa)", pressure_range[0], pressure_range[1], 200, 10),
                    "eh": st.slider("氧化还原电位 (mV)", eh_range[0], eh_range[1], 100),
                    "sulfur_content": st.slider("硫含量 (wt%)", sulfur_range[0], sulfur_range[1], 0.5, 0.01),
                    "chlorine_content": st.slider("氯含量 (wt%)", chlorine_range[0], chlorine_range[1], 5.0, 0.1)
                }
            elif selected_scene == "li_weathering":
                # 核心修复：使用get方法读取water_mobility_range，永不触发KeyError
                mobility_range = current_scene.get("water_mobility_range", (0.1, 10.0))
                additional_params["water_mobility"] = st.slider(
                    "水的流动性（降水和水流）",
                    mobility_range[0],  # 不再用["water_mobility_range"]
                    mobility_range[1],
                    5.0,
                    0.1,
                    help="数值越大，Li流失越快"
                )

            # 模拟时间步长
            time_steps = st.slider("模拟时间步长", 100, 20000, 5000 if selected_scene == "au_hydrothermal" else 10000, 100)

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
                    with st.spinner("模拟中..."):
                        sim = st.session_state.sim
                        scene = st.session_state.current_scene
                        params = st.session_state.params

                        # 设置水流动性（Li场景）
                        if selected_scene == "li_weathering":
                            sim.set_water_mobility(params.get("water_mobility", 1.0))

                        # 执行模拟
                        time_points, avg_concs = [], []
                        solver = sim.explicit_solver if scene.get("solver_type") == "explicit" else sim.implicit_solver
                        diff_coeff = scene.get("diffusion_coeff", 1e-6)
                        reaction_rate = scene.get("reaction_rate", 1e-4)

                        progress = st.progress(0)
                        steps = int(params["time_steps"])
                        for step in range(steps):
                            solver(diff_coeff, reaction_rate)
                            if step % 200 == 0:
                                time_points.append(sim.time)
                                avg_concs.append(np.mean(sim.concentration))
                            progress.progress((step+1)/steps)
                        progress.empty()

                        # 计算结果
                        vis = ResultVisualization(sim)
                        enrichment_factor = vis.calculate_enrichment_factor(scene.get("initial_concentration", 0.01))

                        # 保存结果
                        st.session_state.sim_results = {
                            "enrichment_factor": enrichment_factor,
                            "simulation_time": sim.time,
                            "time_points": time_points,
                            "avg_concentrations": avg_concs,
                            "scene_name": scene.get("name"),
                            "water_mobility": params.get("water_mobility", 1.0)
                        }
                        st.success("模拟完成！")
                except Exception as e:
                    st.error(f"模拟出错：{str(e)}")

    # 右侧结果展示
    st.header("📊 模拟结果")
    if not st.session_state.current_scene:
        st.info("请先加载场景并运行模拟")
    else:
        results = st.session_state.sim_results
        if results:
            # 核心指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_name = "流失系数" if "li_weathering" in selected_scene else "富集系数"
                st.metric(metric_name, f"{results.get('enrichment_factor', 0):.2f}")
            with col2:
                st.metric("总模拟时间", f"{results.get('simulation_time', 0):.0f}")
            with col3:
                st.metric("最高浓度", f"{np.max(st.session_state.sim.concentration):.4f} ppm")
            with col4:
                st.metric("场景名称", results.get('scene_name', '未知'))

            # Li场景显示水流动性
            if "li_weathering" in selected_scene:
                st.metric("水的流动性", f"{results.get('water_mobility', 1.0):.1f}")

            st.divider()

            # 图表展示
            try:
                vis = ResultVisualization(st.session_state.sim)
                tab1, tab2 = st.tabs(["等值线图", "时间曲线"])
                with tab1:
                    st.pyplot(vis.plot_contour())
                with tab2:
                    st.pyplot(vis.plot_time_series(results["time_points"], results["avg_concentrations"]))
            except Exception as e:
                st.error(f"图表出错：{str(e)}")

            st.divider()

            # 数据导出
            st.subheader("💾 数据导出")
            col1, col2 = st.columns(2)
            with col1:
                excel_data = ResultVisualization(st.session_state.sim).export_excel()
                if excel_data:
                    st.download_button(
                        "导出Excel",
                        data=excel_data,
                        file_name=f"{results.get('scene_name', '数据')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            with col2:
                vtk_data = ResultVisualization(st.session_state.sim).export_vtk()
                st.download_button(
                    "导出VTK",
                    data=vtk_data,
                    file_name=f"{results.get('scene_name', '数据')}.vtk",
                    mime="text/plain"
                )

# ===================== 运行程序 =====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"程序出错：{str(e)}")
        st.session_state.clear()
        st.info("请刷新页面重新运行")
