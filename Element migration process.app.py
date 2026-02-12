import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import csv
import json
from io import StringIO, BytesIO
from typing import Dict, List, Tuple
import pandas as pd

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False
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
        self.time += self.dt
        return self.concentration

    def water_rock_reaction(self, mineral_dissolution_rate: float, surface_area: float) -> float:
        """水-岩相互作用：矿物溶解动力学模型"""
        return mineral_dissolution_rate * surface_area * (1 - self.concentration / self.saturation_concentration)

    def magma_crystallization(self, distribution_coefficient: float, melt_fraction: float) -> np.ndarray:
        """岩浆结晶分异：瑞利结晶模型"""
        return self.concentration * (1 - melt_fraction) ** (distribution_coefficient - 1)


# ===================== 2. 场景预设与自定义模块 =====================
class SceneManager:
    """管理内置场景与自定义场景"""

    def __init__(self):
        self.scenes: Dict[str, Dict] = {
            "au_hydrothermal": {
                "name": "热液蚀变Au富集",
                "initial_concentration": 0.01,  # ppm
                "temperature_range": (200, 300),  # ℃
                "ph_range": (4.5, 6.0),
                "time_range": (100, 10000),  # 小时
                "dt": 1.0,  # 时间步长（小时）
                "diffusion_coeff": 1e-6,
                "reaction_rate": 1e-4,
                "solver_type": "explicit"
            },
            "li_weathering": {
                "name": "风化淋滤Li流失",
                "initial_concentration": 50,  # ppm
                "ph_range": (3.0, 5.0),
                "time_range": (1000, 100000),  # 年
                "dt": 100.0,  # 时间步长（年）
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

    def plot_contour(self, title: str = "浓度等值线图") -> plt.Figure:
        """绘制浓度等值线图（返回matplotlib fig对象）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(self.simulation.concentration, cmap='viridis', levels=20)
        plt.colorbar(contour, ax=ax, label='浓度 (ppm)')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('空间坐标X')
        ax.set_ylabel('空间坐标Y')
        plt.tight_layout()
        return fig

    def plot_time_series(self, time_points: List[float], concentrations: List[float],
                         title: str = "浓度-时间曲线") -> plt.Figure:
        """绘制浓度随时间变化曲线"""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_points, concentrations, 'b-', linewidth=2)
        ax.set_xlabel('时间')
        ax.set_ylabel('平均浓度 (ppm)')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def calculate_enrichment_factor(self, initial_concentration: float) -> float:
        """计算元素富集系数"""
        max_concentration = np.max(self.simulation.concentration)
        return max_concentration / initial_concentration if initial_concentration > 0 else 0.0

    def export_csv(self) -> StringIO:
        """导出浓度场数据为CSV（返回内存文件对象）"""
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['X坐标', 'Y坐标', '浓度(ppm)'])
        for i in range(self.simulation.domain_size[0]):
            for j in range(self.simulation.domain_size[1]):
                writer.writerow([i, j, self.simulation.concentration[i, j]])
        output.seek(0)  # 重置文件指针
        return output

    def export_vtk(self) -> StringIO:
        """导出浓度场数据为VTK格式"""
        output = StringIO()
        output.write("# vtk DataFile Version 3.0\n")
        output.write("Geochemical Element Migration Simulation\n")
        output.write("ASCII\n")
        output.write("DATASET STRUCTURED_POINTS\n")
        output.write(f"DIMENSIONS {self.simulation.domain_size[1]} {self.simulation.domain_size[0]} 1\n")
        output.write(f"ORIGIN 0 0 0\n")
        output.write(f"SPACING {self.simulation.dx} {self.simulation.dy} 1\n")
        output.write(f"POINT_DATA {self.simulation.domain_size[0] * self.simulation.domain_size[1]}\n")
        output.write("SCALARS concentration float 1\n")
        output.write("LOOKUP_TABLE default\n")
        for i in range(self.simulation.domain_size[0]):
            for j in range(self.simulation.domain_size[1]):
                output.write(f"{self.simulation.concentration[i, j]:.6f} ")
        output.seek(0)
        return output


# ===================== 4. 教学管理模块 =====================
class TeachingManagement:
    """教学任务管理与数据统计"""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}  # 教学任务库
        self.student_data: Dict[str, List[str]] = {}  # 学生学习数据

    def create_task(self, task_id: str, scene_name: str, param_ranges: Dict, deadline: str) -> None:
        """创建教学实验任务"""
        self.tasks[task_id] = {
            "scene_name": scene_name,
            "param_ranges": param_ranges,
            "deadline": deadline,
            "submissions": {}  # 学生提交记录
        }

    def submit_experiment(self, task_id: str, student_id: str, params: Dict, results: Dict) -> None:
        """学生提交实验报告"""
        if task_id in self.tasks:
            self.tasks[task_id]["submissions"][student_id] = {
                "params": params,
                "results": results,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            # 更新学生学习数据
            if student_id not in self.student_data:
                self.student_data[student_id] = []
            self.student_data[student_id].append(task_id)

    def auto_grade(self, task_id: str, student_id: str) -> Tuple[str, str]:
        """自动批改实验报告"""
        if task_id not in self.tasks or student_id not in self.tasks[task_id]["submissions"]:
            return "错误", "任务或学生不存在"

        submission = self.tasks[task_id]["submissions"][student_id]
        param_ranges = self.tasks[task_id]["param_ranges"]

        # 检查参数是否在允许范围内
        params_valid = all(
            param_ranges[k][0] <= submission["params"][k] <= param_ranges[k][1]
            for k in param_ranges
        )
        # 检查结果合理性（富集系数>1为有效）
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
    # 初始化Streamlit会话状态（保存全局变量）
    if "sim" not in st.session_state:
        st.session_state.sim = NumericalSimulation(domain_size=(50, 50), dx=1.0, dy=1.0, dt=1.0)
    if "scene_manager" not in st.session_state:
        st.session_state.scene_manager = SceneManager()
    if "teaching_manager" not in st.session_state:
        st.session_state.teaching_manager = TeachingManagement()
        # 初始化默认教学任务
        st.session_state.teaching_manager.create_task(
            task_id="GEOCHEM_TASK_001",
            scene_name="au_hydrothermal",
            param_ranges={
                "temperature": (200, 300),
                "ph": (4.5, 6.0),
                "time_steps": (100, 10000)
            },
            deadline="2024-12-31"
        )
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = {}
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = {}
    if "concentration_data" not in st.session_state:
        st.session_state.concentration_data = None

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
            # 初始化浓度场
            st.session_state.sim.concentration = np.full(
                st.session_state.sim.domain_size,
                st.session_state.current_scene["initial_concentration"]
            )
            st.session_state.sim.dt = st.session_state.current_scene["dt"]
            st.success(f"成功加载场景：{st.session_state.current_scene['name']}")

        st.divider()

        # 2. 参数调整（仅当加载场景后显示）
        if st.session_state.current_scene:
            st.subheader("⚙️ 参数调整")
            temperature = st.slider(
                "温度 (℃)",
                min_value=st.session_state.current_scene["temperature_range"][0],
                max_value=st.session_state.current_scene["temperature_range"][1],
                value=int(np.mean(st.session_state.current_scene["temperature_range"]))
            )
            ph = st.slider(
                "pH值",
                min_value=st.session_state.current_scene["ph_range"][0],
                max_value=st.session_state.current_scene["ph_range"][1],
                value=float(np.mean(st.session_state.current_scene["ph_range"])),
                step=0.1
            )
            time_steps = st.slider(
                "模拟时间步长",
                min_value=int(st.session_state.current_scene["time_range"][0] // st.session_state.current_scene["dt"]),
                max_value=int(st.session_state.current_scene["time_range"][1] // st.session_state.current_scene["dt"]),
                value=int(st.session_state.current_scene["time_range"][1] // st.session_state.current_scene["dt"])
            )

            # 保存参数到会话状态
            st.session_state.params = {
                "temperature": temperature,
                "ph": ph,
                "time_steps": time_steps
            }

            # 3. 运行模拟按钮
            if st.button("▶️ 运行模拟"):
                with st.spinner("正在执行数值模拟..."):
                    sim = st.session_state.sim
                    scene = st.session_state.current_scene
                    params = st.session_state.params

                    time_points = []
                    avg_concentrations = []
                    solver = sim.explicit_solver if scene["solver_type"] == "explicit" else sim.implicit_solver

                    # 执行时间步迭代（带进度条）
                    progress_bar = st.progress(0)
                    for step in range(int(params["time_steps"])):
                        solver(scene["diffusion_coeff"], scene["reaction_rate"])
                        # 每100步记录一次数据
                        if step % 100 == 0:
                            time_points.append(sim.time)
                            avg_concentrations.append(np.mean(sim.concentration))
                        # 更新进度条
                        progress_bar.progress((step + 1) / int(params["time_steps"]))

                    # 生成可视化结果
                    vis = ResultVisualization(sim)
                    contour_fig = vis.plot_contour(title=f"{scene['name']} - 浓度等值线图")
                    time_fig = vis.plot_time_series(time_points, avg_concentrations,
                                                    title=f"{scene['name']} - 浓度-时间曲线")

                    # 计算核心指标
                    enrichment_factor = vis.calculate_enrichment_factor(scene["initial_concentration"])

                    # 保存结果到会话状态
                    st.session_state.sim_results = {
                        "contour_fig": contour_fig,
                        "time_fig": time_fig,
                        "enrichment_factor": enrichment_factor,
                        "simulation_time": sim.time,
                        "time_points": time_points,
                        "avg_concentrations": avg_concentrations
                    }
                    # 保存CSV数据
                    st.session_state.concentration_data = vis.export_csv()

                    st.success("模拟完成！结果已展示在主界面")

    # ===== 右侧：结果展示与数据导出 =====
    st.header("📊 模拟结果展示")

    if not st.session_state.current_scene:
        st.info("请先在左侧加载预设场景并运行模拟")
    else:
        # 显示模拟核心指标
        if st.session_state.sim_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("富集系数", f"{st.session_state.sim_results['enrichment_factor']:.2f}")
            with col2:
                st.metric("总模拟时间", f"{st.session_state.sim_results['simulation_time']:.2f}")
            with col3:
                st.metric("场景名称", st.session_state.current_scene["name"])

            st.divider()

            # 显示可视化图表
            tab1, tab2 = st.tabs(["浓度等值线图", "浓度-时间曲线"])
            with tab1:
                st.pyplot(st.session_state.sim_results["contour_fig"])
            with tab2:
                st.pyplot(st.session_state.sim_results["time_fig"])

            st.divider()

            # 数据导出
            st.subheader("💾 数据导出")
            col_csv, col_vtk = st.columns(2)
            with col_csv:
                st.download_button(
                    label="导出CSV数据",
                    data=st.session_state.concentration_data,           
                    file_name=f"{st.session_state.current_scene['name']}.csv",
                    mime="text/csv" 
                )
                
            with col_vtk:
                # 生成VTK数据
                vis = ResultVisualization(st.session_state.sim)
                vtk_data = vis.export_vtk()
                st.download_button(
                    label="导出VTK数据",
                    data=vtk_data,
                    file_name=f"{st.session_state.current_scene['name']}_浓度数据.vtk",
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
                st.write(f"批改结果：{grade}")
                st.write(f"评语：{comment}")

        with col3:
            if st.button("导出统计数据"):
                stats = st.session_state.teaching_manager.export_statistics(task_id)
                if stats:
                    st.write("任务统计数据：")
                    st.json(stats)
                else:
                    st.warning("该任务无统计数据")


if __name__ == "__main__":
    main()


