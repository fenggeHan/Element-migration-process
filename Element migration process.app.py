import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
from typing import Dict, List, Tuple

# ===================== 强制中文字体配置 =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"

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
            (np.roll(self.concentration, -1, axis=1) + np.roll(self.concentration, 1, axis=1) - 2 * self.concentration) / self.dx **2 +
            (np.roll(self.concentration, -1, axis=0) + np.roll(self.concentration, 1, axis=0) - 2 * self.concentration) / self.dy **2
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
            for i in range(1, self.domain_size[0]-1):
                for j in range(1, self.domain_size[1]-1):
                    new_concentration[i, j] = (
                        self.concentration[i, j] + self.dt * diffusion_coeff * (
                            (self.concentration[i+1,j] + self.concentration[i-1,j])/self.dx** 2 +
                            (self.concentration[i,j+1] + self.concentration[i,j-1])/self.dy** 2
                        )
                    ) / (1 + self.dt * (2*diffusion_coeff*(1/self.dx**2 + 1/self.dy** 2) + reaction_rate))
        self.concentration = new_concentration
        self.concentration = np.clip(self.concentration, 0, None)
        self.time += self.dt
        return self.concentration

    def reset_concentration(self):
        self.concentration = np.zeros(self.domain_size)
        self.time = 0.0

# ===================== 2. 场景预设 =====================
class SceneManager:
    def __init__(self):
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
                "ph_range": (3.0, 5.0),
                "pressure_range": (10, 1000),
                "eh_range": (-200, 400),
                "sulfur_content_range": (0.01, 1.0),
                "chlorine_content_range": (0.1, 10.0),
                "dt": 100.0,
                "diffusion_coeff": 1e-7,
                "reaction_rate": 1e-5,
                "solver_type": "implicit"
            }
        }

    def get_scene(self, scene_name: str) -> Dict:
        return self.scenes.get(scene_name, {})

# ===================== 3. 可视化模块 =====================
class ResultVisualization:
    def __init__(self, simulation: NumericalSimulation):
        self.sim = simulation

    def plot_contour(self, title="浓度等值线图"):
        fig, ax = plt.subplots(figsize=(9,7), dpi=150)
        c = self.sim.concentration
        vmin, vmax = np.min(c), np.max(c)
        if vmax - vmin < 1e-6:
            vmax = vmin + 0.02
        cf = ax.contourf(c, cmap="viridis", levels=20, vmin=vmin, vmax=vmax)
        ax.contour(c, colors='white', linewidths=0.5, alpha=0.5)
        plt.colorbar(cf, ax=ax, label="浓度 (ppm)")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("空间坐标 X", fontsize=12)
        ax.set_ylabel("空间坐标 Y", fontsize=12)
        plt.tight_layout()
        return fig

    def plot_time_series(self, time_series, avg_conc, title="浓度-时间变化曲线"):
        fig, ax = plt.subplots(figsize=(10,4), dpi=150)
        ax.plot(time_series, avg_conc, linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("时间", fontsize=12)
        ax.set_ylabel("平均浓度 (ppm)", fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def export_excel(self):
        # 导出 Excel（完全可用）
        x = []
        y = []
        conc = []
        nx, ny = self.sim.domain_size
        for i in range(nx):
            for j in range(ny):
                x.append(i)
                y.append(j)
                conc.append(self.sim.concentration[i,j])
        df = pd.DataFrame({
            "X坐标": x,
            "Y坐标": y,
            "浓度(ppm)": conc
        })
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="浓度数据", index=False)
        buffer.seek(0)
        return buffer

# ===================== 4. 教学管理 =====================
class TeachingManagement:
    def __init__(self):
        self.tasks = {}
        self.student_data = {}

    def create_task(self, task_id, scene_name, param_ranges, deadline):
        self.tasks[task_id] = {
            "scene_name": scene_name,
            "param_ranges": param_ranges,
            "deadline": deadline,
            "submissions": {}
        }

    def submit_experiment(self, task_id, student_id, params, results):
        if task_id in self.tasks:
            self.tasks[task_id]["submissions"][student_id] = {
                "params": params,
                "results": results,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

# ===================== 5. 主界面 =====================
def main():
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation((50,50))
    if "scene_mgr" not in st.session_state:
        st.session_state.scene_mgr = SceneManager()
    if "teach_mgr" not in st.session_state:
        st.session_state.teach_mgr = TeachingManagement()
        st.session_state.teach_mgr.create_task(
            "GEOCHEM_TASK_001", "au_hydrothermal",
            {
                "temperature": (0,1000), "ph":(2,8), "pressure":(10,1000),
                "eh":(-200,400), "sulfur_content":(0.01,1), "chlorine_content":(0.1,10)
            },
            "2026-12-31"
        )
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
        sel_scene = st.selectbox("选择场景", list(scene_opt.keys()), format_func=lambda x: scene_opt[x])

        if st.button("✅ 加载场景", type="primary"):
            sc = st.session_state.scene_mgr.get_scene(sel_scene)
            st.session_state.current_scene = sc
            sim = st.session_state.sim
            sim.reset_concentration()
            c0 = sc["initial_concentration"]
            sim.concentration[:,:] = c0
            cx, cy = sim.domain_size[0]//2, sim.domain_size[1]//2
            sim.concentration[cx-5:cx+5, cy-5:cy+5] = c0 * 10
            sim.dt = sc["dt"]
            st.success(f"已加载：{sc['name']}")

        if st.session_state.current_scene:
            st.subheader("参数设置")
            temp = st.slider("温度 (℃)", 0, 1000, 300)
            ph = st.slider("pH", 2.0, 8.0, 5.0, 0.1)
            add_params = {}
            if sel_scene == "au_hydrothermal":
                press = st.slider("压力 (MPa)", 10, 1000, 200)
                eh = st.slider("Eh (mV)", -200,400,100)
                S = st.slider("硫含量", 0.01,1.0,0.5,0.01)
                Cl = st.slider("氯含量",0.1,10.0,5.0,0.1)
                add_params = {"pressure":press, "eh":eh, "sulfur":S, "chlorine":Cl}
            steps = st.slider("模拟步长", 100, 20000, 5000, 100)

            if st.button("▶️ 运行模拟"):
                sim = st.session_state.sim
                sc = st.session_state.current_scene
                solver = sim.explicit_solver if sc["solver_type"] == "explicit" else sim.implicit_solver
                ts = []
                avgs = []
                bar = st.progress(0)
                for i in range(int(steps)):
                    solver(sc["diffusion_coeff"], sc["reaction_rate"])
                    if i % 200 == 0:
                        ts.append(sim.time)
                        avgs.append(np.mean(sim.concentration))
                    bar.progress((i+1)/steps)
                bar.empty()

                vis = ResultVisualization(sim)
                ef = np.max(sim.concentration) / sc["initial_concentration"] if sc["initial_concentration"]>0 else 0
                st.session_state.sim_results = {
                    "time_series": ts,
                    "avg_conc": avgs,
                    "enrichment": ef,
                    "total_time": sim.time,
                    "scene_name": sc["name"]
                }
                st.success("模拟完成！")

    st.header("📊 结果展示")
    if st.session_state.current_scene and st.session_state.sim_results:
        res = st.session_state.sim_results
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("富集系数", f"{res['enrichment']:.2f}")
        c2.metric("总时间", f"{res['total_time']:.0f}")
        c3.metric("最高浓度", f"{np.max(st.session_state.sim.concentration):.4f} ppm")
        c4.metric("场景", res["scene_name"])

        st.divider()
        vis = ResultVisualization(st.session_state.sim)
        t1, t2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
        with t1:
            fig1 = vis.plot_contour(f"{res['scene_name']} - 浓度等值线图")
            st.pyplot(fig1)
        with t2:
            fig2 = vis.plot_time_series(res["time_series"], res["avg_conc"], f"{res['scene_name']} - 浓度-时间曲线")
            st.pyplot(fig2)

        st.divider()
        st.subheader("💾 导出 Excel 数据（可直接打开）")
        excel_data = vis.export_excel()
        st.download_button(
            label="📥 导出浓度数据为 Excel",
            data=excel_data,
            file_name=f"{res['scene_name']}_浓度数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
