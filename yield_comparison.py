import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# 从custom_yield_rate.py导入process_data函数
from custom_yield_rate import process_data

# 设置图表字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 创建保存图片的文件夹
output_dir = 'yield_comparison_figures'
os.makedirs(output_dir, exist_ok=True)

# 定义各碳源的参数
drG_dict = {'Glucose (Cathermo)': 10,
            'Glucose (Thermo)': 2843.1,
            'Fructose': 2843.1,
            'Pyruvate': 1128.48,
            'Gluconate': 2566.45,
            'Galactose': 2843.1}

ag_dict = {'Glucose (Cathermo)': 227.669811,  ##from outpu_0.13_protein.pkl
           'Glucose (Thermo)': 224.175408,  ##from outpu_2.75.pkl
           'Fructose': 253.532701,  ##from outpu_2.pkl
           'Pyruvate': 273.595279,  ##from outpu_3.25_pyr.pkl
           'Gluconate': 256.060117,  ##from outpu_3.25_glcn.pkl
           'Galactose': 232.677024}  ##from outpu_2.25.pkl

ag_ferm_dict = {'Glucose (Cathermo)': 80.829410,  ##from outpu_0.13_protein.pkl
                'Glucose (Thermo)': 77.017202,  ##from outpu_2.75.pkl
                'Fructose': 104.456448,  ##from outpu_2.pkl
                'Pyruvate': 49.205177,  ##from outpu_3.25_pyr.pkl
                'Gluconate': 103.110871,  ##from outpu_3.25_glcn.pkl
                'Galactose': 130.356135}  ##from outpu_2.25.pkl}

drG_fer_dict = {'Glucose (Cathermo)': 392.09,
                'Glucose (Thermo)': 394.83,
                'Fructose': 403.77,
                'Pyruvate': 67.93,
                'Gluconate': 428.74,
                'Galactose': 378.52}

gamma_dict = {'Glucose (Cathermo)': 24,
              'Glucose (Thermo)': 24,
              'Fructose': 24,
              'Pyruvate': 10,
              'Gluconate': 22,
              'Galactose': 24}

molweight_dict = {'Glucose (Cathermo)': 180,
                  'Glucose (Thermo)': 180,
                  'Fructose': 180,
                  'Pyruvate': 87,
                  'Gluconate': 195,
                  'Galactose': 180}

maintain_dict = {'Glucose (Cathermo)': 11.150546,
                 'Glucose (Thermo)': 11.309159,
                 'Fructose': 11.461692,
                 'Pyruvate': 10.929096,
                 'Gluconate': 9.835886,
                 'Galactose': 11.975169}

maintain_ferm_dict = {'Glucose (Cathermo)': 16.395927,
                      'Glucose (Thermo)': 16.634186,
                      'Fructose': 15.869856,
                      'Pyruvate': 32.569370,
                      'Gluconate': 19.947861,
                      'Galactose': 12.539162}
##gamma_D = 3.7608
gamma_D = 4.7154
gamma_ac = 4

D_max = 4.9 * 26.9882

c_dict = {'Glucose (Cathermo)': 6,
          'Glucose (Thermo)': 6,
          'Fructose': 6,
          'Pyruvate': 3,
          'Gluconate': 6,
          'Galactose': 6}

with open('data.pkl', "rb") as f:
    data = pickle.load(f)
rxns = data[0]


# 定义计算yield的函数
def calculate_yield(miu, carbon_source, drG):
    """
    根据给定的生长率(miu)、碳源和自由能变化(drG)，使用herber_pirt公式计算yield

    Args:
        miu: 生长率(1/h)
        carbon_source: 碳源名称
        drG: 完全氧化的自由能改变(kJ/mol)

    Returns:
        float: 计算得到的yield (g_DW/g_substrate)
    """
    gamma = gamma_dict[carbon_source]
    molweight = molweight_dict[carbon_source]
    c = c_dict[carbon_source]
    # 从ag_dict中获取ag值
    ag = ag_dict[carbon_source]

    # 计算临界点 D_max/(ag + 13.12)
    critical_miu = (D_max - maintain_dict[carbon_source]) / ag

    # 当生长率小于等于临界点时，使用原来的计算方法
    if miu <= critical_miu:
        # 计算a和ms参数 - 对drG取绝对值
        a = (ag / abs(drG)) + (
                gamma_D / gamma)

        ms = maintain_dict[carbon_source] / abs(drG)

        # 计算qs
        qs = a * miu + ms
    else:
        # 生长率大于临界点时，使用混合代谢模式
        # 定义参数
        ag2 = ag
        ag1 = ag_ferm_dict[carbon_source]
        drg2 = abs(drG)
        drg1 = drG_fer_dict[carbon_source]

        # 联立方程求解miu1和miu2
        # 方程1: miu1 + miu2 = miu
        # 方程2: ag2*miu2 + ag1*miu1 + 13.12*miu = D_max
        # 解这个方程组
        # 从方程1得到: miu2 = miu - miu1
        # 代入方程2:
        # ag2*(miu - miu1) + ag1*miu1 + 13.12*miu = D_max
        # (ag1 - ag2)*miu1 = D_max - ag2*miu - 13.12*miu
        # miu1 = (D_max - miu*(ag2 + 13.12)) / (ag1 - ag2)
        denominator = ag1 - ag2
        if denominator == 0:
            # 避免除以零，使用原来的计算方法
            a = (ag / abs(drG)) + (gamma_D / gamma)
            ms = maintain_dict[carbon_source] / abs(drG)
            qs = a * miu + ms
        else:
            miu1 = (D_max - miu * ag2 - maintain_ferm_dict[carbon_source]) / denominator
            miu2 = miu - miu1

            # 确保miu1和miu2都是非负数
            miu1 = max(0, miu1)
            miu2 = max(0, miu2)

            # 计算a1和a2
            a1 = (ag1 / abs(drg1)) + (gamma_D / gamma)
            a2 = (ag2 / abs(drg2)) + (gamma_D / gamma)

            # 计算ms
            ms = miu1 * maintain_ferm_dict[carbon_source] / abs(drg1) + miu2 * maintain_ferm_dict[carbon_source] / abs(
                drg2)

            # 计算qs
            qs = a1 * miu1 + a2 * miu2 + ms

    # 计算yield
    yied = (miu * 26.9882) / (qs * molweight)

    return yied


# 数据加载和保存函数
def load_or_process_simulation_data(data_file='simulation_data_4.0.pkl'):
    """
    加载或处理模拟数据
    如果数据文件存在，则直接加载；否则处理数据并保存

    Args:
        data_file: 数据保存的文件名

    Returns:
        dict: 包含各碳源模拟数据的字典
    """
    import pickle

    # 检查数据文件是否存在
    if os.path.exists(data_file):
        print(f"正在从{data_file}加载模拟数据...")
        with open(data_file, 'rb') as f:
            return pickle.load(f)

    # 如果文件不存在，处理数据
    print("正在处理模拟数据...")

    # 定义各碳源的文件夹和反应名称
    carbon_source_info = {
        'Glucose (Thermo)': {'folder': 'thermo', 'reaction': 'EX_glc', 'pattern': 'thermo'},
        'Fructose': {'folder': 'fru', 'reaction': 'EX_fru', 'pattern': 'fru'},
        'Pyruvate': {'folder': 'pyr', 'reaction': 'EX_pyr', 'pattern': 'pyr'},
        'Gluconate': {'folder': 'glcn', 'reaction': 'EX_glcn', 'pattern': 'glcn'},
        'Galactose': {'folder': 'gal', 'reaction': 'EX_gal', 'pattern': 'gal'}
    }

    # 创建保存数据的字典
    simulation_data = {}

    # 获取并处理模拟数据
    for carbon_source, info in carbon_source_info.items():
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), info['folder'])
        if os.path.exists(folder_path):
            print(f"正在处理 {carbon_source} 的模拟数据...")

            # 加载data.pkl文件获取rxns等信息
            data_path = os.path.join(folder_path, 'data.pkl')
            with open(data_path, "rb") as f:
                folder_data = pickle.load(f)

            # 提取rxns信息
            if isinstance(folder_data, tuple) and len(folder_data) > 0:
                rxns = folder_data[0]
            else:
                rxns = folder_data

            # 获取所需反应的索引
            try:
                carbon_idx = rxns.index(info['reaction'])
                co2_idx = rxns.index('EX_co2')
                h_idx = rxns.index('EX_h')
                h2o_idx = rxns.index('EX_h2o')
                o2_idx = rxns.index('EX_o2')
                acetate_idx = rxns.index('EX_ac')
            except ValueError as e:
                print(f"反应索引获取失败: {e}")
                continue

            # 调用从custom_yield_rate.py导入的process_data函数
            max_growth_data, _ = process_data(folder_path, info['reaction'], info['pattern'])

            if max_growth_data:
                # 计算反应自由能
                # 读取最大生长率对应的output文件中的吉布斯自由能
                # 获取output文件列表
                output_files = []

                # 通用文件查找逻辑
                for file_name in os.listdir(folder_path):
                    if file_name.startswith('output_') and file_name.endswith('.pkl'):
                        file_path = os.path.join(folder_path, file_name)
                        output_files.append(file_path)

                # 如果没有找到文件，使用原来的模式匹配
                if not output_files:
                    if info['pattern'] == 'cathermo':
                        for i in np.arange(0, 1.0, 0.01):
                            file_path = os.path.join(folder_path, f'output_{i:.2f}_protein.pkl')
                            if os.path.exists(file_path):
                                output_files.append(file_path)
                    elif info['pattern'] == 'thermo':
                        for i in np.arange(0.25, 12.1, 0.25):
                            file_path = os.path.join(folder_path, f'output_{i:.2f}.pkl')
                            if os.path.exists(file_path):
                                output_files.append(file_path)
                    else:
                        for i in range(0, 20):
                            file_path = os.path.join(folder_path, f'output_{i}.pkl')
                            if os.path.exists(file_path):
                                output_files.append(file_path)

                # 定义一个函数来从文件名中提取数字部分
                def extract_number(file_path):
                    filename = os.path.basename(file_path)
                    base_name = filename[7:-4]  # 'output_xxx' -> 'xxx'

                    # 移除可能的后缀
                    suffixes = ['_glcn', '_glyc', '_pyr', '_protein', '_ac', '_fru']
                    for suffix in suffixes:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break

                    try:
                        return float(base_name)
                    except ValueError:
                        return 0

                # 对文件进行排序，确保与process_data中使用的排序方式一致
                output_files.sort(key=extract_number)

                # 为每个文件计算其最大生长率点的oxidation_drG
                oxidation_drGs = []
                file_data_list = []  # 存储每个文件的最大生长率、yield和oxidation_drG

                for file_path in output_files:
                    try:
                        with open(file_path, "rb") as file:
                            data = pickle.load(file)

                        # 查找当前文件的最大生长率点
                        file_max_growth = 0
                        file_max_growth_yield = 0
                        file_max_growth_drG = None
                        file_co2_drG = None
                        file_h_drG = None
                        file_h2o_drG = None
                        file_o2_drG = None
                        file_acetate_drG = None
                        ex_carbon_sum = 0
                        k = 0

                        for l in range(len(data)):
                            if hasattr(data[l], 'termin') and hasattr(data[l], 'obj1') and hasattr(data[l],
                                                                                                   'soluDict1'):
                                if (data[l].termin != 'infeasible' and
                                        data[l].termin != 'maxIterations' and
                                        data[l].termin != 'restoration failed'):

                                    current_growth = data[l].obj1

                                    # 计算yield
                                    if carbon_idx < len(data[l].soluDict1['v']):
                                        current_ex_carbon = data[l].soluDict1['v'][carbon_idx]
                                        if current_ex_carbon != 0:
                                            # 对于不同碳源，yield计算需要不同的系数
                                            if info['reaction'] == 'EX_fru':
                                                current_yield = abs(current_growth / current_ex_carbon / 0.18)
                                            elif info['reaction'] == 'EX_glcn':
                                                current_yield = abs(current_growth / current_ex_carbon / 0.196)
                                            elif info['reaction'] == 'EX_pyr':
                                                current_yield = abs(current_growth / current_ex_carbon / 0.088)
                                            else:  # EX_glc, EX_gal
                                                current_yield = abs(current_growth / current_ex_carbon / 0.18)

                                    # 检查是否有足够的drG数据
                                    if (carbon_idx < len(data[l].soluDict1['dfG']) and
                                            co2_idx < len(data[l].soluDict1['dfG']) and
                                            h_idx < len(data[l].soluDict1['dfG']) and
                                            h2o_idx < len(data[l].soluDict1['dfG']) and
                                            o2_idx < len(data[l].soluDict1['dfG']) and
                                            acetate_idx < len(data[l].soluDict1['dfG'])):

                                        # 记录当前文件的最大生长率点和对应的drG
                                        if current_growth > file_max_growth:
                                            file_max_growth = current_growth
                                            file_max_growth_yield = current_yield
                                            file_max_growth_drG = data[l].soluDict1['dfG'][carbon_idx]
                                            file_co2_drG = data[l].soluDict1['dfG'][co2_idx]
                                            file_h_drG = data[l].soluDict1['dfG'][h_idx]
                                            file_h2o_drG = data[l].soluDict1['dfG'][h2o_idx]
                                            file_o2_drG = data[l].soluDict1['dfG'][o2_idx]
                                            file_acetate_drG = data[l].soluDict1['dfG'][acetate_idx]

                        # 计算当前文件的oxidation_drG
                        file_oxidation_drG = None
                        file_fermentation_drG = None
                        if (file_max_growth_drG is not None and file_co2_drG is not None and
                                file_h_drG is not None and file_h2o_drG is not None and file_o2_drG is not None and
                                file_acetate_drG is not None):
                            # 根据不同碳源计算氧化反应的drG
                            if carbon_source in ['Glucose (Cathermo)', 'Glucose (Thermo)', 'Fructose', 'Galactose']:
                                # 6CO2 + 6H+ - 碳源 - 6O2
                                file_oxidation_drG = 6 * file_co2_drG + 6 * file_h_drG - file_max_growth_drG - 6 * file_o2_drG
                                # 3*acetate + 3*h - carbon_source
                                file_fermentation_drG = 3 * file_acetate_drG + 3 * file_h_drG - file_max_growth_drG
                            elif carbon_source == 'Pyruvate':
                                # 3CO2 + 2H+ - 碳源 - 2.5O2 - H2O
                                file_oxidation_drG = 3 * file_co2_drG + 2 * file_h_drG - file_max_growth_drG - 2.5 * file_o2_drG - file_h2o_drG
                                # 4/5*acetate + 1/2*co2 + 1/4*h - carbon_source - 1/2*h2o
                                file_fermentation_drG = (4 / 5) * file_acetate_drG + (1 / 2) * file_co2_drG + (
                                            1 / 4) * file_h_drG - file_max_growth_drG - (1 / 2) * file_h2o_drG
                            elif carbon_source == 'Gluconate':
                                # 6CO2 + 5H+ - 碳源 - 5.5O2
                                file_oxidation_drG = 6 * file_co2_drG + 5 * file_h_drG - file_max_growth_drG - 5.5 * file_o2_drG
                                # 11/4*acetate + 1/2*co2 + 7/4*h + 1/2*h2o - Gluconate
                                file_fermentation_drG = (11 / 4) * file_acetate_drG + (1 / 2) * file_co2_drG + (
                                            9 / 4) * file_h_drG - file_max_growth_drG

                        # 只有当文件有有效的最大生长率时才添加数据
                        if file_max_growth > 0:
                            file_data_list.append(
                                (file_max_growth, file_max_growth_yield, file_oxidation_drG, file_fermentation_drG))

                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

                # 从file_data_list中提取生长率、产量和oxidation_drG列表
                growths = [data[0] for data in file_data_list]
                yields = [data[1] for data in file_data_list]
                oxidation_drGs = [data[2] for data in file_data_list]
                fermentation_drGs = [data[3] for data in file_data_list]

                simulation_data[carbon_source] = {
                    'growths': growths,
                    'yields': yields,
                    'oxidation_drGs': oxidation_drGs,  # 保存每个文件的oxidation_drG
                    'fermentation_drGs': fermentation_drGs  # 保存每个文件的fermentation_drG
                }
                print(f"找到 {len(growths)} 个 {carbon_source} 的模拟数据点")
            else:
                simulation_data[carbon_source] = {
                    'growths': [],
                    'yields': [],
                    'oxidation_drG': None,
                    'fermentation_drG': None
                }
        else:
            print(f"文件夹 {folder_path} 不存在，跳过 {carbon_source}")
            simulation_data[carbon_source] = {
                'growths': [],
                'yields': [],
                'oxidation_drG': None
            }

    # 保存处理后的数据
    with open(data_file, 'wb') as f:
        pickle.dump(simulation_data, f)
    print(f"模拟数据已保存到{data_file}")

    return simulation_data


# 函数：绘制不同碳源的理论yield和miu的图，同时包含模拟数据
def plot_yield_theory_and_simulation():
    """
    绘制不同碳源生长率从0到0.7的理论yield曲线，并与模拟数据进行比较
    """
    # 定义生长率范围
    miu_range = np.linspace(0.01, 0.7, 100)

    # 定义碳源颜色和标记
    carbon_sources = ['Glucose (Thermo)', 'Fructose', 'Pyruvate', 'Gluconate',
                      'Galactose']
    colors = ['darkgreen', 'blue', 'orange', 'purple', 'magenta']
    markers = ['^', 'o', 's', 'd', 'p']

    # 创建图形
    plt.figure(figsize=(6, 4))

    # 获取模拟数据（使用加载或处理函数）
    simulation_data = load_or_process_simulation_data()

    # 绘制不同碳源的理论yield点和模拟数据点
    for i, carbon_source in enumerate(carbon_sources):
        if carbon_source in simulation_data:
            data = simulation_data[carbon_source]
            growths = data.get('growths', [])
            yields = data.get('yields', [])
            oxidation_drGs = data.get('oxidation_drGs', [])

            # 确保三个列表长度一致
            min_length = min(len(growths), len(yields), len(oxidation_drGs))
            growths = growths[:min_length]
            yields = yields[:min_length]
            oxidation_drGs = oxidation_drGs[:min_length]

            # 计算有效oxidation_drGs的平均值
            valid_drGs = [drG for drG in oxidation_drGs if drG is not None]
            if valid_drGs:
                average_drG = np.mean(valid_drGs)
                print(f"{carbon_source} 的平均drG值: {average_drG}")

                # 使用平均drG在整个生长率范围内计算理论yield曲线
                theory_yields = [calculate_yield(miu, carbon_source, average_drG) for miu in miu_range]

                # 处理碳源名称，将'Glucose (Cathermo)'和'Glucose (Thermo)'改为'Glucose'
                display_name = carbon_source.replace('Glucose (Cathermo)', 'Glucose').replace('Glucose (Thermo)',
                                                                                              'Glucose')

                # 绘制理论yield线，减小线宽
                plt.plot(miu_range, theory_yields, color=colors[i], linewidth=1.5,
                         label=f'{display_name} (Theory)')

            # 绘制模拟数据点，减小点大小并设为空心点
            if growths and yields:
                # 处理碳源名称，将'Glucose (Cathermo)'和'Glucose (Thermo)'改为'Glucose'
                display_name = carbon_source.replace('Glucose (Cathermo)', 'Glucose').replace('Glucose (Thermo)',
                                                                                              'Glucose')
                plt.scatter(growths, yields, color=colors[i], marker=markers[i], s=50,
                            label=f'{display_name} (Simulation)')

    # 添加标题和标签

    plt.xlabel('growth rate (1/h)')
    plt.ylabel('yield (g$_{DW}$/g$_{substrate}$)')

    # 添加图例，放在图外
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    # 设置坐标轴范围
    plt.xlim(0, 0.75)
    plt.ylim(0.15, None)  # 自动调整y轴上限

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(output_dir, 'yield_theory_vs_simulation_6.0.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图片已保存: yield_theory_vs_simulation_5.0.pdf")




# 主函数
def plot_combined_figures():
    """
    为每个碳源绘制包含两个子图的图表：
    1. 碳源摄入率与生长率关系图
    2. 吉布斯自由能耗散随生长率变化图
    """
    # 定义碳源列表
    carbon_sources = ['Glucose (Thermo)', 'Fructose', 'Pyruvate', 'Gluconate',
                      'Galactose']

    # 获取模拟数据
    simulation_data = load_or_process_simulation_data()

    # 加载calculated_ag_data.pkl文件
    try:
        with open('calculated_ag_data.pkl', 'rb') as f:
            calculated_ag_data = pickle.load(f)
        print("成功加载calculated_ag_data.pkl文件")
    except Exception as e:
        print(f"加载calculated_ag_data.pkl文件失败: {e}")
        calculated_ag_data = {}

    # 为每个碳源单独绘制图表
    for carbon_source in carbon_sources:
        
        display_name = carbon_source.replace('Glucose (Cathermo)', 'Glucose').replace('Glucose (Thermo)', 'Glucose')

        # 定义生长率范围
        miu_range = np.linspace(0.01, 0.7, 100)

        # 创建一个包含两个子图的figure，每个子图大小为3x2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))

        # 第一个子图：碳源摄入率与生长率关系
        if carbon_source in simulation_data:
            data = simulation_data[carbon_source]
            growths = data.get('growths', [])
            yields = data.get('yields', [])
            oxidation_drGs = data.get('oxidation_drGs', [])

            # 确保三个列表长度一致
            min_length = min(len(growths), len(yields), len(oxidation_drGs))
            growths = growths[:min_length]
            yields = yields[:min_length]
            oxidation_drGs = oxidation_drGs[:min_length]

            # 计算有效oxidation_drGs的平均值
            valid_drGs = [drG for drG in oxidation_drGs if drG is not None]
            if valid_drGs:
                average_drG = np.mean(valid_drGs)

                # 计算理论碳源摄入率qs、生长率miu和无氧呼吸生长率miu1
                theory_miu = miu_range
                theory_qs = []
                theory_miu1 = []  # 无氧呼吸产生的生长率
                molweight = molweight_dict.get(carbon_source, 180)  # 默认葡萄糖分子量
                ag = ag_dict[carbon_source]
                ag1 = ag_ferm_dict[carbon_source]

                for miu in miu_range:
                    yield_val = calculate_yield(miu, carbon_source, average_drG)
                    # 理论碳摄入率：(miu/yield)/molweight
                    if yield_val > 0:
                        qs = (miu / yield_val) / molweight
                        # 转换单位
                        qs = qs * 1000
                    else:
                        qs = 0
                    theory_qs.append(qs)

                    # 计算miu1（无氧呼吸生长率）
                    critical_miu = (D_max - maintain_dict[carbon_source]) / ag
                    if miu <= critical_miu:
                        miu1_val = 0
                    else:
                        denominator = ag1 - ag
                        if denominator == 0:
                            miu1_val = 0
                        else:
                            miu1_val = (D_max - miu * ag - maintain_ferm_dict[carbon_source]) / denominator
                            miu1_val = max(0, miu1_val)
                    theory_miu1.append(miu1_val)

                # 绘制理论总生长率曲线（cyan颜色）
                ax1.plot(theory_qs, theory_miu, color='cyan', linewidth=1.0,
                         label='Growth Rate (Theory)')

                # 绘制miu1曲线（红色）
                ax1.plot(theory_qs, theory_miu1, color='red', linewidth=1.0, linestyle='-',
                         label='Fermentation Growth Rate (Theory)')

                # 为miu1下方区域涂色（红色）
                ax1.fill_between(theory_qs, 0, theory_miu1, color='red', alpha=0.3,
                                 label='Fermentation Contribution')

            # 计算模拟数据的碳源摄入率current_ex_carbon
            if growths and yields:
                sim_qs = []
                sim_miu = []
                molweight = molweight_dict.get(carbon_source, 180)  # 默认葡萄糖分子量
                for growth, yield_val in zip(growths, yields):
                    if yield_val > 0:
                        # 模拟碳摄入率：growth / (yield_val * (molweight/1000))，不用转换单位
                        current_ex_carbon = growth / (yield_val * (molweight / 1000))
                        sim_qs.append(current_ex_carbon)
                        sim_miu.append(growth)

                # 绘制模拟数据点
                ax1.scatter(sim_qs, sim_miu, color='blue', marker='o', s=10, facecolors='none',
                            label='Growth Rate (Simulation)')

                # 添加miu1_values的数据点
                if carbon_source in calculated_ag_data:
                    miu1_values = calculated_ag_data.get(carbon_source, {}).get('miu1_values', [])
                    growth_values = calculated_ag_data.get(carbon_source, {}).get('growth_rates', [])
                    if miu1_values and sim_miu and sim_qs:
                        # 为miu1_values从simulation_data中查找对应的current_ex_carbon
                        miu1_qs = []
                        miu1_valid = []

                        for growth in growth_values:
                            # 在sim_miu中找到最接近的生长率
                            if growth in sim_miu:
                                # 找到最接近miu1的生长率索引
                                index = sim_miu.index(growth)
                                index2 = growth_values.index(growth)
                                miu1_qs.append(sim_qs[index])
                                miu1_valid.append(miu1_values[index2])

                        if miu1_qs:
                            # 绘制miu1_values数据点
                            ax1.scatter(miu1_qs, miu1_valid, color='green', marker='s', s=10, facecolors='none',
                                        label='Fermentation Growth Rate (Simulation)')

        # 添加标题和标签
        ax1.set_title(display_name)
        ax1.set_xlabel(' Substrate uptake rate (g/g$_{DW}$/h)')
        ax1.set_ylabel('Growth rate (1/h)')

        # 设置坐标轴范围
        ax1.set_xlim(0, None)  # 自动调整x轴上限
        ax1.set_ylim(0, 0.75)

        # 第二个子图：吉布斯自由能耗散随生长率变化
        # 初始化存储数组
        total_dissipation = []
        aerobic_dissipation = []
        anaerobic_dissipation = []

        # 获取碳源参数
        ag = ag_dict[carbon_source]
        ag1 = ag_ferm_dict[carbon_source]
        drG = drG_dict[carbon_source]
        drg1 = drG_fer_dict[carbon_source]
        gamma = gamma_dict[carbon_source]

        # 计算临界点
        critical_miu = (D_max - maintain_dict[carbon_source]) / ag

        for miu in miu_range:
            if miu <= critical_miu:
                # 只进行有氧代谢
                a = (ag / abs(drG)) + (gamma_D / gamma)
                ms = maintain_dict[carbon_source] / abs(drG)
                qs = a * miu + ms

                # 计算有氧代谢的自由能耗散
                aerobic = ag * miu + maintain_dict[carbon_source]
                anaerobic = 0
            else:
                # 混合代谢模式
                denominator = ag1 - ag
                if denominator == 0:
                    # 避免除以零，使用原来的计算方法
                    a = (ag / abs(drG)) + (gamma_D / gamma)
                    ms = maintain_dict[carbon_source] / abs(drG)
                    qs = a * miu + ms

                    aerobic = ag * miu + maintain_dict[carbon_source]
                    anaerobic = 0
                else:
                    # 计算miu1和miu2
                    miu1 = (D_max - miu * ag - maintain_ferm_dict[carbon_source]) / denominator
                    miu1 = max(0, miu1)
                    miu2 = max(0, miu - miu1)

                    # 计算有氧和无氧代谢的自由能耗散
                    aerobic = ag * miu2 + maintain_ferm_dict[carbon_source] * (miu2 / miu) if miu > 0 else 0
                    anaerobic = ag1 * miu1 + maintain_ferm_dict[carbon_source] * (miu1 / miu) if miu > 0 else 0

            # 计算总自由能耗散
            total = aerobic + anaerobic

            total_dissipation.append(total)
            aerobic_dissipation.append(aerobic)
            anaerobic_dissipation.append(anaerobic)

        # 绘制总自由能耗散曲线（深蓝色）
        ax2.plot(miu_range, total_dissipation, color='darkblue', linewidth=1.0)

        # 绘制respiration曲线（红色）
        ax2.plot(miu_range, anaerobic_dissipation, color='red', linewidth=1.0)

        # 为有氧和无氧部分涂色
        ax2.fill_between(miu_range, 0, anaerobic_dissipation, color='red', alpha=0.3, label='fermentation')
        ax2.fill_between(miu_range, anaerobic_dissipation, total_dissipation, color='mediumslateblue', alpha=0.6,
                         label='respirition')

        # 添加标题和标签
        ax2.set_title(display_name)
        ax2.set_xlabel('Growth rate (1/h)')
        ax2.set_ylabel('Gibbs energy dissipation (kJ/g$_{DW}$/h)')

        # 设置坐标轴范围
        ax2.set_xlim(0, 0.75)
        ax2.set_ylim(0, None)  # 自动调整y轴上限

        # 调整布局
        plt.tight_layout()

        # 保存合并后的图表为PDF
        # 替换碳源名称中的空格和括号，以便文件名更规范
        filename = carbon_source.replace(' (', '_').replace(')', '').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'combined_{filename}_5.0.pdf'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"图片已保存: combined_{filename}_5.0.pdf")


def main():
    """主函数，执行所有绘图功能"""
    print("开始绘制理论与模拟产率比较图...")
    plot_yield_theory_and_simulation()


    print("所有图表绘制完成！")


if __name__ == "__main__":
    main()