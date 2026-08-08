import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 创建保存图片的文件夹
output_dir = 'custom_rate_yield_figures'
os.makedirs(output_dir, exist_ok=True)

# 设置图表字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 函数：从不同文件夹读取数据并计算yield
def process_data(folder_path, carbon_source, file_pattern):
    """
    处理指定文件夹的数据
    
    Args:
        folder_path: 文件夹路径
        carbon_source: 碳源名称 ('EX_glc' 或 'EX_fru'等)
        file_pattern: 文件命名模式 ('thermo', 'cathermo', 'fru')
    
    Returns:
        tuple: (file_max_growth_data, all_optimal_data)
        file_max_growth_data格式: (max_growth, max_yield, avg_ex_carbon, carbon_drG)
    """
    # 加载data.pkl文件获取rxns等信息
    data_path = os.path.join(folder_path, 'data.pkl')
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # 提取rxns信息
    if isinstance(data, tuple) and len(data) > 0:
        rxns = data[0]
    else:
        rxns = data
    
    # 获取所需反应的索引
    carbon_idx = rxns.index(carbon_source)
    biomass_idx = rxns.index('biomass')
    
    # 获取output文件列表
    output_files = []
    
    # 通用文件查找逻辑，支持处理带有后缀的文件名
    # 直接遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.startswith('output_') and file_name.endswith('.pkl'):
            # 支持各种后缀格式，如_glcn、_glyc、_pyr、_ac、_fru等
            file_path = os.path.join(folder_path, file_name)
            output_files.append(file_path)
    
    # 如果没有找到文件，使用原来的模式匹配
    if not output_files:
        if file_pattern == 'cathermo':
            # cathermo文件夹的output_X.XX_protein.pkl文件
            for i in np.arange(0, 1.0, 0.01):
                file_path = os.path.join(folder_path, f'output_{i:.2f}_protein.pkl')
                if os.path.exists(file_path):
                    output_files.append(file_path)
        elif file_pattern == 'thermo':
            # thermo文件夹的output_X.XX.pkl文件
            for i in np.arange(0.25, 12.1, 0.25):
                file_path = os.path.join(folder_path, f'output_{i:.2f}.pkl')
                if os.path.exists(file_path):
                    output_files.append(file_path)
        else:  # 包括fru、glcn、glyc、pyr等
            # 通用整数索引文件模式
            for i in range(0, 20):  # 增加范围到20以确保捕获所有文件
                file_path = os.path.join(folder_path, f'output_{i}.pkl')
                if os.path.exists(file_path):
                    output_files.append(file_path)
    
    # 定义一个函数来从文件名中提取数字部分
    def extract_number(file_path):
        filename = os.path.basename(file_path)
        # 移除 'output_' 前缀和 '.pkl' 后缀
        base_name = filename[7:-4]  # 'output_xxx' -> 'xxx'
        
        # 移除可能的后缀，如 '_glcn', '_glyc', '_pyr', '_protein', '_ac', '_fru' 等
        suffixes = ['_glcn', '_glyc', '_pyr', '_protein', '_ac', '_fru']
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # 尝试转换为浮点数或整数
        try:
            return float(base_name)
        except ValueError:
            # 如果转换失败，返回0作为默认值
            return 0
    
    # 对文件进行排序
    output_files.sort(key=extract_number)
    
    # 存储每个文件的最大生长率点
    file_max_growth_data = []  # 格式: (max_growth, max_yield, avg_ex_carbon)
    
    # 存储每个EX_carbon值下所有optimal解
    all_optimal_data = {}  # 格式: {ex_carbon_value: [(growth, yield)]}
    
    # 处理每个output文件
    for file_path in output_files:
        try:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
            
            # 记录当前文件的最大生长率、平均EX_carbon和碳源反应drG
            file_max_growth = 0
            file_max_growth_yield = 0
            file_max_growth_drG = 0
            ex_carbon_sum = 0
            k = 0
            
            # 存储当前生长率下的所有optimal解
            current_optimal_points = []
            
            for l in range(len(data)):
                if hasattr(data[l], 'termin') and hasattr(data[l], 'obj1') and hasattr(data[l], 'soluDict1'):
                    if (data[l].termin != 'infeasible' and 
                        data[l].termin != 'maxIterations' and 
                        data[l].termin != 'restoration failed'):
                        
                        current_growth = data[l].obj1
                        
                        if carbon_idx < len(data[l].soluDict1['v']):
                            current_ex_carbon = data[l].soluDict1['v'][carbon_idx]
                            
                            # 计算yield，对于fru数据使用biomass除以EX_fru
                            if current_ex_carbon != 0:
                                # 对于不同碳源，yield计算需要不同的系数
                                if carbon_source == 'EX_fru':
                                    # fructose的分子量约为180.16 g/mol
                                    current_yield = abs(current_growth / current_ex_carbon / 0.18)
                                elif carbon_source == 'EX_glcn':
                                    # gluconate除以0.196
                                    current_yield = abs(current_growth / current_ex_carbon / 0.196)
                                elif carbon_source == 'EX_pyr':
                                    # pyruvate除以0.088
                                    current_yield = abs(current_growth / current_ex_carbon / 0.088)
                                elif carbon_source == 'EX_glyc':
                                    # glycerol除以0.092
                                    current_yield = abs(current_growth / current_ex_carbon / 0.092)
                                else:  # EX_glc
                                    # glucose的分子量约为180.16 g/mol
                                    current_yield = abs(current_growth / current_ex_carbon / 0.18)
                                
                                # 添加到当前optimal解列表
                                current_optimal_points.append((current_growth, current_yield))
                                
                                # 记录当前文件的最大生长率点和对应的drG
                                if current_growth > file_max_growth:
                                    file_max_growth = current_growth
                                    file_max_growth_yield = current_yield
                                    # 获取碳源反应的drG值
                                    if carbon_idx < len(data[l].soluDict1['dfG']):
                                        file_max_growth_drG = data[l].soluDict1['dfG'][carbon_idx]
                                
                                ex_carbon_sum += abs(current_ex_carbon)  # 使用绝对值，因为摄取通常是负值
                                k += 1
            
            if k > 0:
                ex_carbon_avg = ex_carbon_sum / k
                
                # 保存当前EX_carbon下的所有optimal解
                all_optimal_data[ex_carbon_avg] = current_optimal_points
                
                # 保存当前文件的数据：最大生长率点、平均EX_carbon和碳源反应drG
                file_max_growth_data.append((file_max_growth, file_max_growth_yield, ex_carbon_avg, file_max_growth_drG))
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return file_max_growth_data, all_optimal_data

# 函数：绘制第四张图（类似CA_ThermoFBA的图，包含cathermo和fru数据）


# 函数：绘制第五张图（类似ThermoFBA的图，包含thermo和fru数据）


# 函数：将三组数据（cathermo、thermo和fru）显示在同一张图里


# 函数：将所有数据源数据保存为CSV文件
def save_all_data_to_csv(cathermo_data, thermo_data, fru_data, pyr_data, glcn_data, glyc_data, gal_data):
    """
    将所有数据源的生长率、产量和碳源反应drG数据保存为CSV文件
    
    Args:
        cathermo_data: cathermo数据源数据
        thermo_data: thermo数据源数据
        fru_data: fru数据源数据
        pyr_data: pyr数据源数据
        glcn_data: glcn数据源数据
        glyc_data: glyc数据源数据
        gal_data: gal数据源数据
    """
    # 创建数据列表
    data_list = []
    
    # 添加cathermo数据
    for point in cathermo_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Cathermo (Glucose)',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加thermo数据
    for point in thermo_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Thermo (Glucose)',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加fru数据
    for point in fru_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Fructose',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加pyr数据
    for point in pyr_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Pyruvate',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加glcn数据
    for point in glcn_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Gluconate',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加glyc数据
    for point in glyc_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Glycerol',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 添加gal数据
    for point in gal_data:
        growth_rate, yield_value, avg_ex_carbon, drG = point
        data_list.append({
            'source': 'Galactose',
            'growth_rate': growth_rate,
            'yield': yield_value,
            'avg_ex_carbon': avg_ex_carbon,
            'carbon_drG': drG
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存为CSV文件
    csv_path = os.path.join(output_dir, 'all_sources_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"所有数据源数据已保存到CSV文件: {csv_path}")

# 函数：将所有数据源（cathermo、thermo、fru、pyr、glcn、glyc）显示在同一张图里（归一化版本）
def plot_normalized_all_sources(cathermo_data, thermo_data, fru_data, pyr_data, glcn_data, glyc_data, gal_data):
    """
    将所有数据源的生长率和产量数据归一化后显示在同一张图里
    归一化基准：每个碳源中yield最大的点
    
    Args:
        cathermo_data: cathermo数据源数据
        thermo_data: thermo数据源数据
        fru_data: fru数据源数据
        pyr_data: pyr数据源数据
        glcn_data: glcn数据源数据
        glyc_data: glyc数据源数据
        gal_data: gal数据源数据
    """
    plt.figure(figsize=(14, 12))
    
    # 定义一个函数来获取和归一化数据
    def get_normalized_data(data):
        # 找到yield最大的点
        if not data:
            return [], []
        
        max_yield_point = max(data, key=lambda x: x[1])
        max_yield = max_yield_point[1]
        
        # 确保max_yield不为0，避免除零错误
        if max_yield == 0:
            max_yield = 1
        
        # 提取和归一化数据
        growths = [point[0] / max_yield for point in data]  # 生长率除以最大yield
        yields = [point[1] / max_yield for point in data]   # yield除以最大yield
        
        return growths, yields
    
    # 获取各数据源的归一化数据
    cathermo_growths, cathermo_yields = get_normalized_data(cathermo_data)
    thermo_growths, thermo_yields = get_normalized_data(thermo_data)
    fru_growths, fru_yields = get_normalized_data(fru_data)
    pyr_growths, pyr_yields = get_normalized_data(pyr_data)
    glcn_growths, glcn_yields = get_normalized_data(glcn_data)
    glyc_growths, glyc_yields = get_normalized_data(glyc_data)
    gal_growths, gal_yields = get_normalized_data(gal_data)
    
    # 绘制各数据源数据（保持与原函数相同的颜色和标记）
    plt.scatter(cathermo_growths, cathermo_yields, color='red', marker='*', s=150, 
                label='Cathermo (Glucose)')
    
    plt.scatter(thermo_growths, thermo_yields, color='darkgreen', marker='^', s=120, 
                label='Thermo (Glucose)')
    
    plt.scatter(fru_growths, fru_yields, color='blue', marker='o', s=100, 
                label='Fructose')
    
    plt.scatter(pyr_growths, pyr_yields, color='orange', marker='s', s=100, 
                label='Pyruvate')
    
    plt.scatter(glcn_growths, glcn_yields, color='purple', marker='d', s=100, 
                label='Gluconate')
    
    plt.scatter(glyc_growths, glyc_yields, color='cyan', marker='x', s=100, 
                label='Glycerol')
    
    plt.scatter(gal_growths, gal_yields, color='magenta', marker='p', s=100, 
                label='Galactose')
    
    # 添加标题和标签
    plt.title('All Carbon Sources: Normalized Growth Rate vs Normalized Yield', fontweight='bold')
    plt.xlabel('Normalized growth rate (w.r.t maximum yield point)', fontweight='bold')
    plt.ylabel('Normalized yield (w.r.t maximum yield point)', fontweight='bold')
    
    # 添加图例
    plt.legend(fontsize=14, prop={'weight': 'bold'}, loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'normalized_all_sources.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("归一化所有数据源图已保存: normalized_all_sources.png")

# 函数：将所有数据源（cathermo、thermo、fru、pyr、glcn、glyc）显示在同一张图里
def plot_yield_with_CmoldfG(cathermo_data, thermo_data, fru_data, pyr_data, glcn_data, glyc_data, gal_data):
    """
    绘制所有碳源的生长率与yield乘以对应碳源CmoldfG值的关系图
    
    Args:
        cathermo_data: cathermo数据源数据
        thermo_data: thermo数据源数据
        fru_data: fructose数据源数据
        pyr_data: pyruvate数据源数据
        glcn_data: gluconate数据源数据
        glyc_data: glycerol数据源数据
        gal_data: galactose数据源数据
    """
    plt.figure(figsize=(14, 12))
    
    # 定义各碳源的CmoldfG值
    #cmoldfG_values = {
     #   'EX_glc': 69.52,    # glucose
     #   'EX_fru': 69.54,    # fructose
     #   'EX_gal': 70.44,    # galactose
     #   'EX_pyr': 117.13,   # pyruvate
     #   'EX_glcn': 112.09,  # gluconate
     #   'EX_glyc': 54.08    # glycerol
    #}


    cmoldfG_values = {
        'EX_glc': 473.85,    # glucose
        'EX_fru': 473.85,    # fructose
        'EX_gal': 473.85,    # galactose
        'EX_pyr': 376.16,   # pyruvate
        'EX_glcn': 427.74,  # gluconate
        'EX_glyc': 542.94    # glycerol
    }
    
    # 提取各数据源数据并计算修正后的yield
    # glucose (cathermo)
    cathermo_growths = [point[0] for point in cathermo_data]
    cathermo_yields = [point[1]/cmoldfG_values['EX_glc'] for point in cathermo_data]
    
    # glucose (thermo)
    thermo_growths = [point[0] for point in thermo_data]
    thermo_yields = [point[1] / cmoldfG_values['EX_glc'] for point in thermo_data]
    
    # fructose
    fru_growths = [point[0] for point in fru_data]
    fru_yields = [point[1] / cmoldfG_values['EX_fru'] for point in fru_data]
    
    # pyruvate
    pyr_growths = [point[0] for point in pyr_data]
    pyr_yields = [point[1] / cmoldfG_values['EX_pyr'] for point in pyr_data]
    
    # gluconate
    glcn_growths = [point[0] for point in glcn_data]
    glcn_yields = [point[1] / cmoldfG_values['EX_glcn'] for point in glcn_data]
    
    # glycerol
    glyc_growths = [point[0] for point in glyc_data]
    glyc_yields = [point[1] / cmoldfG_values['EX_glyc'] for point in glyc_data]
    
    # galactose
    gal_growths = [point[0] for point in gal_data]
    gal_yields = [point[1] / cmoldfG_values['EX_gal'] for point in gal_data]
    
    # 绘制各数据源数据
    plt.scatter(cathermo_growths, cathermo_yields, color='red', marker='*', s=150, 
                label='Cathermo (Glucose)')
    
    plt.scatter(thermo_growths, thermo_yields, color='darkgreen', marker='^', s=120, 
                label='Thermo (Glucose)')
    
    plt.scatter(fru_growths, fru_yields, color='blue', marker='o', s=100, 
                label='Fructose')
    
    plt.scatter(pyr_growths, pyr_yields, color='orange', marker='s', s=100, 
                label='Pyruvate')
    
    plt.scatter(glcn_growths, glcn_yields, color='purple', marker='d', s=100, 
                label='Gluconate')
    
    plt.scatter(glyc_growths, glyc_yields, color='cyan', marker='x', s=100, 
                label='Glycerol')
    
    plt.scatter(gal_growths, gal_yields, color='magenta', marker='+', s=100, 
                label='Galactose')
    
    # 添加标题和标签
    plt.title('Growth Rate vs Yield * CmoldfG for All Carbon Sources', fontweight='bold')
    plt.xlabel('Growth Rate (h^-1)', fontweight='bold')
    plt.ylabel('Yield * CmoldfG (g biomass / g carbon)', fontweight='bold')
    
    # 添加图例
    plt.legend(fontsize=14, prop={'weight': 'bold'}, loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'yield_with_CmoldfG.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Yield乘以CmoldfG的图表已保存: yield_with_CmoldfG.png")

def plot_all_sources(cathermo_data, thermo_data, fru_data, pyr_data, glcn_data, glyc_data, gal_data):
    plt.figure(figsize=(14, 12))
    
    # 提取各数据源数据
    cathermo_growths = [point[0] for point in cathermo_data]
    cathermo_yields = [point[1] for point in cathermo_data]
    
    thermo_growths = [point[0] for point in thermo_data]
    thermo_yields = [point[1] for point in thermo_data]
    
    fru_growths = [point[0] for point in fru_data]
    fru_yields = [point[1] for point in fru_data]
    
    pyr_growths = [point[0] for point in pyr_data]
    pyr_yields = [point[1] for point in pyr_data]
    
    glcn_growths = [point[0] for point in glcn_data]
    glcn_yields = [point[1] for point in glcn_data]
    
    glyc_growths = [point[0] for point in glyc_data]
    glyc_yields = [point[1] for point in glyc_data]
    
    gal_growths = [point[0] for point in gal_data]
    gal_yields = [point[1] for point in gal_data]
    
    # 绘制各数据源数据
    plt.scatter(cathermo_growths, cathermo_yields, color='red', marker='*', s=150, 
                label='Cathermo (Glucose)')
    
    plt.scatter(thermo_growths, thermo_yields, color='darkgreen', marker='^', s=120, 
                label='Thermo (Glucose)')
    
    plt.scatter(fru_growths, fru_yields, color='blue', marker='o', s=100, 
                label='Fructose')
    
    plt.scatter(pyr_growths, pyr_yields, color='orange', marker='s', s=100, 
                label='Pyruvate')
    
    plt.scatter(glcn_growths, glcn_yields, color='purple', marker='d', s=100, 
                label='Gluconate')
    
    plt.scatter(glyc_growths, glyc_yields, color='cyan', marker='x', s=100, 
                label='Glycerol')
    
    plt.scatter(gal_growths, gal_yields, color='magenta', marker='p', s=100, 
                label='Galactose')
    
    # 添加标题和标签
    plt.title('All Carbon Sources: Growth Rate vs Yield', fontweight='bold')
    plt.xlabel('growth rate(1/h)', fontweight='bold')
    plt.ylabel('yield (g$_{DW}$/g$_{substrate}$)', fontweight='bold')
    
    # 添加图例
    plt.legend(fontsize=14, prop={'weight': 'bold'}, loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'all_sources_growth_points.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("所有碳源数据图已保存: all_sources_growth_points.png")

# 主函数
def main():
    # 定义文件夹路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cathermo_dir = os.path.join(base_dir, 'cathermo')
    thermo_dir = os.path.join(base_dir, 'thermo')
    fru_dir = os.path.join(base_dir, 'fru')
    pyr_dir = os.path.join(base_dir, 'pyr')
    glcn_dir = os.path.join(base_dir, 'glcn')
    glyc_dir = os.path.join(base_dir, 'glyc')
    gal_dir = os.path.join(base_dir, 'gal')
    
    print("开始处理cathermo数据...")
    cathermo_max_growth_data, _ = process_data(cathermo_dir, 'EX_glc', 'cathermo')
    print(f"找到 {len(cathermo_max_growth_data)} 个cathermo文件的数据点")
    
    print("开始处理thermo数据...")
    thermo_max_growth_data, _ = process_data(thermo_dir, 'EX_glc', 'thermo')
    print(f"找到 {len(thermo_max_growth_data)} 个thermo文件的数据点")
    
    print("开始处理fru数据...")
    fru_max_growth_data, _ = process_data(fru_dir, 'EX_fru', 'fru')
    print(f"找到 {len(fru_max_growth_data)} 个fru文件的数据点")
    
    print("开始处理pyr数据...")
    pyr_max_growth_data, _ = process_data(pyr_dir, 'EX_pyr', 'pyr')
    print(f"找到 {len(pyr_max_growth_data)} 个pyr文件的数据点")
    
    print("开始处理glcn数据...")
    glcn_max_growth_data, _ = process_data(glcn_dir, 'EX_glcn', 'glcn')
    print(f"找到 {len(glcn_max_growth_data)} 个glcn文件的数据点")
    
    print("开始处理glyc数据...")
    glyc_max_growth_data, _ = process_data(glyc_dir, 'EX_glyc', 'glyc')
    print(f"找到 {len(glyc_max_growth_data)} 个glyc文件的数据点")
    
    print("开始处理gal数据...")
    gal_max_growth_data, _ = process_data(gal_dir, 'EX_gal', 'gal')
    print(f"找到 {len(gal_max_growth_data)} 个gal文件的数据点")
    

    
    # 绘制包含所有文件夹数据的图表
    plot_all_sources(cathermo_max_growth_data, thermo_max_growth_data, fru_max_growth_data, 
                    pyr_max_growth_data, glcn_max_growth_data, glyc_max_growth_data, gal_max_growth_data)
    
    # 绘制归一化的所有数据源图表
    plot_normalized_all_sources(cathermo_max_growth_data, thermo_max_growth_data, fru_max_growth_data, 
                              pyr_max_growth_data, glcn_max_growth_data, glyc_max_growth_data, gal_max_growth_data)
    
    # 绘制yield乘以CmoldfG的图表
    plot_yield_with_CmoldfG(cathermo_max_growth_data, thermo_max_growth_data, fru_max_growth_data, 
                          pyr_max_growth_data, glcn_max_growth_data, glyc_max_growth_data, gal_max_growth_data)
    
    # 保存所有数据源数据到CSV文件
    save_all_data_to_csv(cathermo_max_growth_data, thermo_max_growth_data, fru_max_growth_data, 
                        pyr_max_growth_data, glcn_max_growth_data, glyc_max_growth_data, gal_max_growth_data)
    
    print("所有图表已生成并保存到文件夹：", output_dir)

if __name__ == "__main__":
    main()