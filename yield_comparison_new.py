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
drG_dict = {'Glucose (Cathermo)': 2843.1,
            'Glucose (Thermo)': 2843.1,
            'Fructose': 2843.1,
            'Pyruvate': 1128.48,
            'Gluconate': 2566.45,
            'Galactose': 2843.1}

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

##gamma_D = 3.7608
gamma_D = 4.7154

D_max = 4.9 * 26.9882

c_dict = {'Glucose (Cathermo)': 6,
          'Glucose (Thermo)': 6,
          'Fructose': 6,
          'Pyruvate': 3,
          'Gluconate': 6,
          'Galactose': 6}

# 定义drG_fer_dict (来自yield_comparison_5.0.py)
drG_fer_dict = {'Glucose (Cathermo)': 310.62,
                'Glucose (Thermo)': 310.62,
                'Fructose': 310.62,
                'Pyruvate': 75.6859,
                'Gluconate': 247.3145,
                'Galactose': 310.62}

with open('data.pkl', "rb") as f:
    data = pickle.load(f)
rxns = data[0]


# 定义计算yield的函数
def calculate_yield(miu, carbon_source):
    """
    根据给定的生长率(miu)、碳源，使用herber_pirt公式计算yield

    Args:
        miu: 生长率(1/h)
        carbon_source: 碳源名称

    Returns:
        float: 计算得到的yield (g_DW/g_substrate)
    """
    gamma = gamma_dict[carbon_source]
    molweight = molweight_dict[carbon_source]
    c = c_dict[carbon_source]
    
    # 使用3.0版本的ag计算公式
    ag2 = (200 + 18 * (6 - c) ** 1.8 + np.exp(((3.8 - gamma / c) ** 2) ** 0.16 * (3.6 + 0.4 * c)))
    print(f"Calculated ag2 for {carbon_source}: {ag2}")
    
    # 计算ag1 = ag2 - 120
    ag1 = ag2 - 120
    
    # 计算critical_miu
    critical_miu = (D_max - 13.12) / ag2
    print(f"Critical miu for {carbon_source}: {critical_miu}")

    # 获取drG值
    drg2 = drG_dict[carbon_source]
    drg1 = drG_fer_dict[carbon_source]

    # 当生长率小于等于临界点时，使用原来的计算方法
    if miu <= critical_miu:
        # 计算a和ms参数 - 对drG取绝对值
        a = (ag2 / abs(drg2)) + (gamma_D / gamma)
        ms = 13.12 / abs(drg2)

        # 计算qs
        qs = a * miu + ms
    else:
        # 生长率大于临界点时，使用混合代谢模式

        # 联立方程求解miu1和miu2
        # 方程1: miu1 + miu2 = miu
        # 方程2: ag2*miu2 + ag1*miu1 + 11*miu = D_max
        denominator = ag1 - ag2
        if denominator == 0:
            # 避免除以零，使用原来的计算方法
            a = (ag2 / abs(drg2)) + (gamma_D / gamma)
            ms = 13.12 / abs(drg2)
            qs = a * miu + ms
        else:
            miu1 = (D_max - miu * ag2 - 13.12) / denominator
            miu2 = miu - miu1

            # 确保miu1和miu2都是非负数
            miu1 = max(0, miu1)
            miu2 = max(0, miu2)

            # 计算a1和a2
            a1 = (ag1 / abs(drg1)) + (gamma_D / gamma)
            a2 = (ag2 / abs(drg2)) + (gamma_D / gamma)

            # 计算ms
            ms = 13.12 / abs(drg2)

            # 计算qs
            qs = a1 * miu1 + a2 * miu2 + ms

    # 计算yield
    yied = (miu * 26.9882) / (qs * molweight)

    return yied


# 数据加载和保存函数 (从3.0版本复制)
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
        'Glucose (Cathermo)': {'folder': 'cathermo', 'reaction': 'EX_glc', 'pattern': 'cathermo'},
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
                                            o2_idx < len(data[l].soluDict1['dfG'])):

                                        # 记录当前文件的最大生长率点和对应的drG
                                        if current_growth > file_max_growth:
                                            file_max_growth = current_growth
                                            file_max_growth_yield = current_yield
                                            file_max_growth_drG = data[l].soluDict1['dfG'][carbon_idx]
                                            file_co2_drG = data[l].soluDict1['dfG'][co2_idx]
                                            file_h_drG = data[l].soluDict1['dfG'][h_idx]
                                            file_h2o_drG = data[l].soluDict1['dfG'][h2o_idx]
                                            file_o2_drG = data[l].soluDict1['dfG'][o2_idx]

                        # 计算当前文件的oxidation_drG
                        file_oxidation_drG = None
                        if (file_max_growth_drG is not None and file_co2_drG is not None and
                                file_h_drG is not None and file_h2o_drG is not None and file_o2_drG is not None):
                            # 根据不同碳源计算氧化反应的drG
                            if carbon_source in ['Glucose (Cathermo)', 'Glucose (Thermo)', 'Fructose', 'Galactose']:
                                # 6CO2 + 6H+ - 碳源 - 6O2
                                file_oxidation_drG = 6 * file_co2_drG + 6 * file_h_drG - file_max_growth_drG - 6 * file_o2_drG
                            elif carbon_source == 'Pyruvate':
                                # 3CO2 + 2H+ - 碳源 - 2.5O2 - H2O
                                file_oxidation_drG = 3 * file_co2_drG + 2 * file_h_drG - file_max_growth_drG - 2.5 * file_o2_drG - file_h2o_drG
                            elif carbon_source == 'Gluconate':
                                # 6CO2 + 5H+ - 碳源 - 5.5O2
                                file_oxidation_drG = 6 * file_co2_drG + 5 * file_h_drG - file_max_growth_drG - 5.5 * file_o2_drG

                        # 只有当文件有有效的最大生长率时才添加数据
                        if file_max_growth > 0:
                            file_data_list.append((file_max_growth, file_max_growth_yield, file_oxidation_drG))

                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

                # 从file_data_list中提取生长率、产量和oxidation_drG列表
                growths = [data[0] for data in file_data_list]
                yields = [data[1] for data in file_data_list]
                oxidation_drGs = [data[2] for data in file_data_list]

                simulation_data[carbon_source] = {
                    'growths': growths,
                    'yields': yields,
                    'oxidation_drGs': oxidation_drGs  # 保存每个文件的oxidation_drG
                }
                print(f"找到 {len(growths)} 个 {carbon_source} 的模拟数据点")
            else:
                simulation_data[carbon_source] = {
                    'growths': [],
                    'yields': [],
                    'oxidation_drG': None
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
    miu_range = np.linspace(0.01, 1.0, 200)  # 扩大范围以显示临界点之后的变化

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
                
                # 使用drG_dict中的值计算理论yield曲线
                theory_yields = [calculate_yield(miu, carbon_source) for miu in miu_range]

                display_name = carbon_source.replace('Glucose (Cathermo)', 'Glucose').replace('Glucose (Thermo)',
                                                                                              'Glucose')
                
                # 绘制理论yield线
                plt.plot(miu_range, theory_yields, color=colors[i], linewidth=1.5,
                         label=f'{display_name} (Theory)')

            # 绘制模拟数据点
            if growths and yields:
                display_name = carbon_source.replace('Glucose (Cathermo)', 'Glucose').replace('Glucose (Thermo)',
                                                                                              'Glucose')
                plt.scatter(growths, yields, color=colors[i], marker=markers[i], s=50,
                            label=f'{display_name} (Simulation)')

    # 添加标题和标签

    plt.xlabel('growth rate (1/h)')
    plt.ylabel('yield (g$_{DW}$/g$_{substrate}$)')

    # 添加图例
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    # 设置坐标轴范围
    plt.xlim(0, 0.75)  # 扩大x轴范围以显示临界点之后的变化
    plt.ylim(0, None)  # 自动调整y轴上限

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(output_dir, 'yield_theory_vs_simulation_new.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图片已保存: yield_theory_vs_simulation_new.pdf")


# 主函数
def main():
    print("开始绘制理论yield曲线和模拟数据比较图...")
    plot_yield_theory_and_simulation()
    print("所有图表已生成并保存到文件夹：", output_dir)


if __name__ == "__main__":
    main()