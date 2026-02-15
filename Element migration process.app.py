import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import csv
from io import StringIO
from typing import Dict, List
import pandas as pd

# ===================== 全局配置 =====================
# 深度优化跨平台字体配置（解决中文显示问题）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.size"] = 11  # 优化字体大小
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示
plt.rcParams["figure.dpi"] = 150  # 提升图片清晰度
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"  # 避免透明背景导致的显示问题
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

# ===================== 3. 结果可视化与分析模块 =====================
class ResultVisualization:
    """结果可视化与分析工具（适配Streamlit）"""

    def __init__(self, simulation: NumericalSimulation):
        self.simulation = simulation

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        """重构等值线图绘制逻辑，确保显示正常"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor="white")

        min_c = np.min(self.simulation.concentration)
        max_c = np.max(self.simulation.concentration)
        if max_c - min_c < 1e-6:  # 浓度无差异时，手动添加层级
            levels = np.linspace(min_c, min_c + 0.02, 20)
        else:
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

        cbar = plt.colorbar(contour, ax=ax, label='浓度 (ppm)', shrink=0.8)
        cbar.ax.set_ylabel('浓度 (ppm)', fontsize=10)

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('空间坐标X', fontsize=12)
        ax.set_ylabel('空间坐标Y', fontsize=12)

        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()

        return fig

    def export_csv(self) -> StringIO:
        """导出浓度场数据为CSV"""
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['X坐标', 'Y坐标', '浓度(ppm)'])
        for i in range(self.simulation.domain_size[0]):
            for j in range(self.simulation.domain_size[1]):
                writer.writerow([i, j, self.simulation.concentration[i, j]])
        output.seek(0)
        return output

# ===================== Streamlit 主逻辑 =====================
def main():
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation(domain_size=(50, 50), dx=1.0, dy=1.0, dt=1.0)

    # ===== 页面标题与布局 =====
    st.title("🌍 地球化学元素迁移虚拟仿真平台")

    # ===== 左侧：场景选择与参数配置 =====
    with st.sidebar:
        st.header("🔧 实验配置")
        selected_scene_key = st.selectbox("选择预设场景", options=["au_hydrothermal", "li_weathering"])

        # 模拟时间步长（100-20000）
        time_steps = st.slider("模拟时间步长", min_value=100, max_value=20000, value=5000, step=100)
        st.session_state.params = {"time_steps": time_steps}

        if st.button("▶️ 运行模拟"):
            with st.spinner("正在执行数值模拟..."):
                time_points = []
                avg_concentrations = []
                sim = st.session_state.sim

                for step in range(int(time_steps)):
                    sim.explicit_solver(1e-6, 1e-4)
                    if step % 200 == 0:
                        time_points.append(sim.time)
                        avg_concentrations.append(np.mean(sim.concentration))
                st.success("模拟完成！结果已展示在主界面")

                vis = ResultVisualization(sim)
                contour_fig = vis.plot_contour(title="浓度等值线图")
                time_fig = vis.plot_contour(title="浓度-时间曲线")

                # 生成可视化结果
                st.session_state.sim_results = {
                    "contour_fig": contour_fig,
                    "time_fig": time_fig
                }
                st.session_state.concentration_data = vis.export_csv()

    # ===== 右侧：结果展示 =====
    st.header("📊 模拟结果展示")
    if st.session_state.sim_results:
        # 可视化图表
        tab1, tab2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
        with tab1:
            st.pyplot(st.session_state.sim_results["contour_fig"], clear_figure=True)
        with tab2:
            st.pyplot(st.session_state.sim_results["time_fig"], clear_figure=True)

        st.subheader("💾 数据导出")
        st.download_button(
            label="导出CSV数据",
            data=st.session_state.concentration_data,
            file_name="浓度数据.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
