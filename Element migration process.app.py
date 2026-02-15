import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO, BytesIO
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib
import platform

# ===================== 全局配置 =====================
def setup_chinese_font():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    system = platform.system()
    font_paths = {
        'Windows': ['SimHei', 'Microsoft YaHei'],
        'Linux': ['WenQuanYi Micro Hei', 'DejaVu Sans'],
        'Darwin': ['PingFang SC', 'Heiti TC']
    }
    candidate_fonts = font_paths.get(system, ['DejaVu Sans'])
    available_fonts = [f for f in candidate_fonts if f in plt.rcParams['font.sans-serif']]
    if available_fonts:
        plt.rcParams["font.family"] = available_fonts
    else:
        plt.rcParams["font.family"] = ['DejaVu Sans']
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

    def central_difference_x(self, field: np.ndarray) -> np.ndarray:
        return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * self.dx)

    def central_difference_y(self, field: np.ndarray) -> np.ndarray:
        return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * self.dy)

    def forward_difference_time(self, field: np.ndarray, rate: np.ndarray) -> np.ndarray:
        return field + rate * self.dt

    def explicit_solver(self, diffusion_coeff: float, reaction_rate: float) -> np.ndarray:
        laplacian = (
            (np.roll(self.concentration, -1, axis=1) + np.roll(self.concentration, 1, axis=1) - 2 * self.concentration) / self.dx ** 2 +
            (np.roll(self.concentration, -1, axis=0) + np.roll(self.concentration, 1, axis=0) - 2 * self.concentration) / self.dy ** 2
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
                    new_concentration[i, j] = (
                        self.concentration[i, j] + self.dt * diffusion_coeff * (
                            (self.concentration[i+1,j] + self.concentration[i-1,j])/self.dx**2 +
                            (self.concentration[i,j+1] + self.concentration[i,j-1])/self.dy**2
                        )
                    ) / (1 + self.dt * (2*diffusion_coeff*(1/self.dx**2 + 1/self.dy**2) + reaction_rate))
        self.concentration = new_concentration
        self.concentration = np.clip(self.concentration, 0, None)
        self.time += self.dt
        return new_concentration

    def reset_concentration(self):
        self.concentration = np.zeros(self.domain_size)
        self.time = 0.0

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
                "ph_range": (3.0, 5.0),
                "temperature_range": (0, 1000),
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
        return self.scenes.get(scene_name, {})

# ===================== 3. 结果可视化与导出 =====================
class ResultVisualization:
    def __init__(self, simulation: NumericalSimulation):
        self.simulation = simulation

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor="white")
        fig.suptitle("Concentration Contour Map", fontsize=14, fontweight='bold')

        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        if max_c - min_c < 1e-6:
            levels = np.linspace(min_c, min_c + 0.02, 20)
        else:
            levels = np.linspace(min_c, max_c, 20)

        contour = ax.contourf(self.simulation.concentration, levels=levels, cmap='viridis', alpha=0.8)
        ax.contour(self.simulation.concentration, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
        cbar = fig.colorbar(contour, ax=ax, label='Concentration (ppm)', shrink=0.8)
        ax.set_xlabel('Spatial Coordinate X', fontsize=12)
        ax.set_ylabel('Spatial Coordinate Y', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_time_series(self, time_points: List[float], concentrations: List[float]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
        ax.plot(time_points, concentrations, 'b-', linewidth=2)
        ax.set_title("Concentration-Time Curve", fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Average Concentration (ppm)', fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def calculate_enrichment_factor(self, initial_concentration: float) -> float:
        max_c = np.max(self.simulation.concentration)
        return max_c / initial_concentration if initial_concentration > 0 else 0.0

    def export_excel(self) -> BytesIO:
        x, y, c = [], [], []
        nx, ny = self.simulation.domain_size
        for i in range(nx):
            for j in range(ny):
                x.append(i)
                y.append(j)
                c.append(self.simulation.concentration[i, j])
        df = pd.DataFrame({"X坐标": x, "Y坐标": y, "浓度(ppm)": c})
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="浓度数据", index=False)
        buffer.seek(0)
        return buffer

# ===================== 4. 主界面 =====================
def main():
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation((50,50))
    if "scene_mgr" not in st.session_state:
        st.session_state.scene_mgr = SceneManager()
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = None
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None

    st.title("🌍 地球化学元素迁移虚拟仿真平台")
    st.divider()

    with st.sidebar:
        st.header("🔧 实验配置")
        scene_opt = {
            "au_hydrothermal": "热液蚀变Au富集",
            "li_weathering": "风化淋滤Li流失"
        }
        sel_scene = st.selectbox("选择预设场景", list(scene_opt.keys()), format_func=lambda x: scene_opt[x])

        if st.button("加载场景", type="primary"):
            sc = st.session_state.scene_mgr.get_scene(sel_scene)
            st.session_state.current_scene = sc
            sim = st.session_state.sim
            sim.reset_concentration()
            c0 = sc["initial_concentration"]
            sim.concentration[:,:] = c0
            cx, cy = sim.domain_size[0]//2, sim.domain_size[1]//2
            sim.concentration[cx-5:cx+5, cy-5:cy+5] = c0 * 10
            sim.dt = sc["dt"]
            st.session_state.sim_results = None
            st.success(f"已加载：{sc['name']}")

        if st.session_state.current_scene:
            st.subheader("⚙️ 参数调整")
            temp = st.slider("温度 (℃)", 0, 1000, 300, 10)
            ph = st.slider("pH值", 2.0, 8.0, 5.0, 0.1)
            add_params = {}
            if sel_scene == "au_hydrothermal":
                press = st.slider("压力 (MPa)", 10, 1000, 200, 10)
                eh = st.slider("氧化还原电位 (mV)", -200,400,100)
                S = st.slider("硫含量", 0.01,1.0,0.5,0.01)
                Cl = st.slider("氯含量",0.1,10.0,5.0,0.1)
                add_params = {"pressure":press, "eh":eh, "sulfur":S, "chlorine":Cl}

            steps = st.slider("模拟时间步长", 100, 20000, 5000, 100)

            if st.button("▶️ 运行模拟"):
                sim = st.session_state.sim
                sc = st.session_state.current_scene
                solver = sim.explicit_solver if sc["solver_type"] == "explicit" else sim.implicit_solver
                ts, avgs = [], []
                bar = st.progress(0)
                for i in range(int(steps)):
                    solver(sc["diffusion_coeff"], sc["reaction_rate"])
                    if i % 200 == 0:
                        ts.append(sim.time)
                        avgs.append(np.mean(sim.concentration))
                    bar.progress((i+1)/steps)
                bar.empty()

                vis = ResultVisualization(sim)
                ef = vis.calculate_enrichment_factor(sc["initial_concentration"])
                st.session_state.sim_results = {
                    "time_series": ts,
                    "avg_conc": avgs,
                    "enrichment": ef,
                    "total_time": sim.time,
                    "scene_name": sc["name"]
                }
                st.success("模拟完成！")

    st.header("📊 模拟结果展示")
    if st.session_state.current_scene and st.session_state.sim_results:
        res = st.session_state.sim_results
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("富集系数", f"{res['enrichment']:.2f}")
        c2.metric("总模拟时间", f"{res['total_time']:.0f}")
        c3.metric("最高浓度", f"{np.max(st.session_state.sim.concentration):.4f} ppm")
        c4.metric("场景", res['scene_name'])

        st.divider()
        vis = ResultVisualization(st.session_state.sim)
        t1, t2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
        with t1:
            st.pyplot(vis.plot_contour())
        with t2:
            st.pyplot(vis.plot_time_series(res["time_series"], res["avg_conc"]))

        st.divider()
        st.subheader("💾 数据导出")
        col1, col2 = st.columns(2)
        with col1:
            excel_buf = vis.export_excel()
            st.download_button(
                "导出 Excel 数据",
                data=excel_buf,
                file_name=f"{res['scene_name']}_浓度数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
