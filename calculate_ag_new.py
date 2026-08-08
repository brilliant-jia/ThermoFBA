import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体为Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 线性拟合函数
def linear_fit(x, y):
    """线性拟合，返回斜率、截距和R²值"""
    if len(x) < 2 or len(y) < 2:
        return None, None, None
    
    # 执行线性拟合
    slope, intercept = np.polyfit(x, y, 1)
    
    # 计算R²值
    y_pred = slope * np.array(x) + intercept
    ss_res = np.sum((np.array(y) - y_pred) ** 2)
    ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return slope, intercept, r_squared

with open('data.pkl', "rb") as f:
    data = pickle.load(f)
rxns = data[0]
drG_dict = {'Glucose (Cathermo)': 2843.1,
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

drG_fer_dict = {'Glucose (Cathermo)': 310.62,
            'Glucose (Thermo)': 310.62,
            'Fructose': 310.62,
            'Pyruvate': 75.6859,
            'Gluconate': 247.3145,
            'Galactose': 310.62}

gamma_dict = {'Glucose (Cathermo)': 24,
              'Glucose (Thermo)': 24,
              'Fructose': 24,
              'Pyruvate': 10,
              'Gluconate': 22,
              'Galactose': 24}
c_dict = {'Glucose (Cathermo)': 6,
          'Glucose (Thermo)': 6,
          'Fructose': 6,
          'Pyruvate': 3,
          'Gluconate': 6,
          'Galactose': 6}

maintain_dict = {'Glucose (Cathermo)': 11.150546,
          'Glucose (Thermo)': 11.309159,
          'Fructose': 11.461692,
          'Pyruvate': 10.929096,
          'Gluconate': 9.835886,
          'Galactose': 11.975169}
gamma_D = 4.7154

D_max = 4.9 * 26.9882

def calculate_ag_for_file(file_path, carbon_source):
    """
    计算单个文件的最大生长率和ag值
    
    参数:
        file_path: str, pkl文件路径
        carbon_source: str, 碳源名称
    
    返回:
        max_growth: float, 最大生长率
        ag: float, 计算得到的ag值
        ag1: float, 无氧呼吸的ag值
        ag2: float, 有氧呼吸的ag值
        miu1: float, 无氧生长率
        miu2: float, 有氧生长率
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        max_growth = 0
        drg = 0
        max_index = 0
        ag1 = None
        ag2 = None
        miu1 = None
        miu2 = None
        
        # 寻找最大生长率的索引
        for i in range(len(data)):
            if data[i].termin == 'optimal':
                if data[i].obj1 > max_growth:
                    max_index = i
                    max_growth = data[i].obj1
        
        # 计算drG*v之和
        if max_growth > 0:  # 确保找到有效解
            for key in data[max_index].soluDict1['drG'].keys():
                if key in data[max_index].soluDict1['v']:
                    drg += data[max_index].soluDict1['drG'][key] * data[max_index].soluDict1['v'][key]
        
        # 计算ag值
        if max_growth > 0:  # 确保有有效的生长率
            ag =  26.9882 * (drg) / (max_growth )

            
            # 检查生长率是否大于临界值D_max/(ag+13.12)
            critical_miu = (D_max -maintain_dict[carbon_source]) / (ag_dict[carbon_source] )
            
            if max_growth > critical_miu:
                # 两种呼吸方式计算
                ag2 = ag_dict[carbon_source] * 1000  # 有氧呼吸的ag值

                
                # 获取drg2和drg1
                drg2 = drG_dict[carbon_source]
                drg1 = drG_fer_dict[carbon_source]
                
                # 读取CO2的绝对通量值vco2
                if rxns.index('EX_o2') in data[max_index].soluDict1['v']:
                    vco2 = abs(data[max_index].soluDict1['v'][rxns.index('EX_o2')])
                else:
                    vco2 = 0
                # 计算有氧生长率miu2
                if vco2 > 0 and drg2 != 0:
                    miu2 = ( ((vco2 / c_dict[carbon_source]) * drg2)/(ag2*41.6/1000 + maintain_dict[carbon_source]/max_growth))
                    miu1 = max_growth - miu2
                    
                    # 确保miu1为正数
                    if miu1 > 0:
                        # 计算ag1
                         ag1  =  (ag*max_growth +miu2*ag2 )/miu1
        else:
            ag = None
        
        return max_growth, ag, ag1, ag2, miu1, miu2
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None, None, None

def main():
    """
    主函数，处理不同碳源的文件并输出结果，绘制不同碳源ag随生长率变化的图
    """
    # 检查数据文件是否存在
    data_file = 'calculated_ag_data.pkl'
    all_data = {}
    
    # 尝试加载数据文件
    if os.path.exists(data_file):
        print(f"发现数据文件 {data_file}，正在加载...")
        loaded_data = load_calculated_ag_data(data_file)
        if loaded_data:
            print("数据加载完成，直接使用数据进行绘图...")
            # 从加载的数据中提取all_data
            carbon_sources = loaded_data.get('carbon_sources', [])
            for carbon_source in carbon_sources:
                if carbon_source in loaded_data:
                    all_data[carbon_source] = loaded_data[carbon_source]
            # 继续执行绘图部分
        else:
            print("数据加载失败，开始进行计算...")
    else:
        print("数据文件不存在，开始进行计算...")
    
    # 定义不同碳源的信息
    carbon_source_info = {
        'Glucose (Cathermo)': {
            'folder_path': 'cathermo',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        },
        'Glucose (Thermo)': {
            'folder_path': 'thermo',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        },

        'Fructose': {
            'folder_path': 'fru',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        },
        'Pyruvate': {
            'folder_path': 'pyr',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        },
        'Gluconate': {
            'folder_path': 'glcn',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        },
        'Galactose': {
            'folder_path': 'gal',
            'file_prefix': 'output_',
            'file_suffix': '.pkl'
        }

    }
    # 如果all_data为空（即没有加载数据），则开始计算
    if not all_data:
        print("开始进行计算...")
        # 处理每个碳源
        for carbon_source, info in carbon_source_info.items():
            print(f"\n=== 处理碳源: {carbon_source} ===")
            
            folder_path = info['folder_path']
            file_prefix = info['file_prefix']
            file_suffix = info['file_suffix']
            
            # 获取目录下所有符合条件的文件
            if not os.path.exists(folder_path):
                print(f"警告: 碳源 {carbon_source} 的目录 {folder_path} 不存在")
                continue
            
            # 收集所有符合条件的文件
            output_files = []
            for filename in os.listdir(folder_path):
                if filename.startswith(file_prefix) and filename.endswith(file_suffix):
                    output_files.append(os.path.join(folder_path, filename))
            
            # 定义一个函数来从文件名中提取数字部分
            def extract_number(file_path):
                filename = os.path.basename(file_path)
                base_name = filename[len(file_prefix):-len(file_suffix)]
                
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
            
            # 对文件进行排序
            output_files.sort(key=extract_number)
            
            # 处理每个文件
            if not output_files:
                print(f"未找到碳源 {carbon_source} 的文件")
                continue
            
            # 收集该碳源的数据
            growth_rates = []
            ag_values = []
            ag1_values = []
            ag2_values = []
            miu1_values = []
            miu2_values = []
            
            print(f"找到 {len(output_files)} 个文件")
            print(f"{'文件名':<20} {'最大生长率':<15} {'ag值':<15} {'ag1值':<15} {'ag2值':<15} {'miu1值':<15} {'miu2值':<15}")
            print("=" * 105)
            
            for file_path in output_files:
                filename = os.path.basename(file_path)
                max_growth, ag, ag1, ag2, miu1, miu2 = calculate_ag_for_file(file_path, carbon_source)
                
                if max_growth is not None and ag is not None:
                    # 格式化输出，处理可能为None的ag1、ag2、miu1和miu2
                    ag1_str = f"{ag1:.6f}" if ag1 is not None else "-"
                    ag2_str = f"{ag2:.6f}" if ag2 is not None else "-"
                    miu1_str = f"{miu1:.6f}" if miu1 is not None else "-"
                    miu2_str = f"{miu2:.6f}" if miu2 is not None else "-"
                    print(f"{filename:<20} {max_growth:<15.6f} {ag:<15.6f} {ag1_str:<15} {ag2_str:<15} {miu1_str:<15} {miu2_str:<15}")
                    
                    # 收集数据
                    growth_rates.append(max_growth)
                    ag_values.append(ag)
                    ag1_values.append(ag1)
                    ag2_values.append(ag2)
                    miu1_values.append(miu1)
                    miu2_values.append(miu2)
                else:
                    print(f"{filename:<20} {'无效':<15} {'无效':<15} {'无效':<15} {'无效':<15} {'无效':<15} {'无效':<15}")
            
            # 保存该碳源的数据
            all_data[carbon_source] = {
                'growth_rates': growth_rates,
                'ag_values': ag_values,
                'ag1_values': ag1_values,
                'ag2_values': ag2_values,
                'miu1_values': miu1_values,
                'miu2_values': miu2_values
            }
    
    # 绘制不同碳源ag随生长率变化的图
    print("\n\n=== 绘制不同碳源ag随生长率变化的图 ===")
    
    # 创建保存图表的文件夹
    save_folder = 'ag_growth_rate_plots'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 绘制ag随生长率变化的图
    plt.figure(figsize=(12, 8))
    
    # 定义不同碳源的颜色
    colors = {
        'Glucose (Cathermo)': 'red',
        'Glucose (Thermo)': 'blue',
        'Fructose': 'green',
        'Pyruvate': 'purple',
        'Gluconate': 'orange',
        'Galactose': 'cyan',
        'Glycerol': 'magenta'
    }
    
    for carbon_source, data in all_data.items():
        if data['growth_rates'] and data['ag_values']:
            # 将ag_values除以1000
            ag_values_scaled = [ag / 1000 for ag in data['ag_values']]
            plt.plot(data['growth_rates'], ag_values_scaled, 'o-', 
                    label=carbon_source, color=colors.get(carbon_source, 'black'))
    

    plt.xlabel('$\mu$ (1/h)')
    plt.ylabel('D (kJ/cmol)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_folder, 'ag_vs_growth_rate_2.0.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    
    # 关闭图表
    plt.close()
    
    # 定义需要拟合的碳源及其数据点范围
    carbon_source_fit_params = {
        'Fructose': {'range': slice(0, 4)},  # 前四个数据点
        'Glucose (Thermo)': {'range': slice(0, 22)},  # 前22个数据点
        'Glucose (Cathermo)': {'range': slice(5, None)},  # 除去前五个数据点剩下的数据点
        'Pyruvate': {'range': slice(0, 3)},  # 前三个数据点
        'Galactose': {'range': slice(0, 4)},  # 前四个数据点
        'Gluconate': {'range': slice(0, 5)}  # 前五个数据点
    }
    
    # 打印拟合结果标题
    print("\n=== 对特定碳源数据点进行线性拟合 (ag vs 1/生长率) ===")
    
    
    # 分别绘制每个碳源的ag随生长率变化的图
    for carbon_source, data in all_data.items():
        if data['growth_rates'] and data['ag_values']:
            fig = plt.figure(figsize=(3, 2))
            ax = fig.add_subplot(111)
            # 将ag_values除以1000
            ag_values_scaled = [ag / 1000 for ag in data['ag_values']]
            ax.plot(data['growth_rates'], ag_values_scaled, 'o-', 
                    label=carbon_source, color=colors.get(carbon_source, 'black'))
            # 设置标题字体
            ax.set_title(f'{carbon_source} ag值随生长率变化图', fontsize=10, fontweight='bold')
            # 设置横纵坐标标签字体
            ax.set_xlabel('$\mu$ (1/h)', fontsize=10, fontweight='bold')
            ax.set_ylabel('D (kJ/cmol)', fontsize=10, fontweight='bold')
            # 设置坐标轴刻度字体
            ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=8)
            ax.tick_params(axis='both', which='minor', width=1.5, length=4)
            ax.grid(True, linestyle='--', alpha=0.7)
            # 设置图例字体
            ax.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(save_folder, f'ag_vs_growth_rate_{carbon_source.replace(" ", "_").replace("(", "").replace(")", "")}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{carbon_source} 图表已保存到: {save_path}")
            
            # 关闭图表
            plt.close()
            
            # Plot ag (as D) vs 1/growthrate for each carbon source (with linear fit)
            fig = plt.figure(figsize=(3, 2))
            ax = fig.add_subplot(111)
            # Calculate 1/growthrate, excluding zero growth rates
            reciprocal_growth_rates = [1/gr if gr > 0 else float('inf') for gr in data['growth_rates']]
            # Convert ag_values to D by dividing by 1000
            D_values = [ag / 1000 for ag in data['ag_values']]
            
            # 处理碳源名称，将"Glucose (Thermo)"替换为"Glucose"
            display_name = carbon_source.replace("Glucose (Thermo)", "Glucose")
            
            # Plot only pre-overflow data points
            if carbon_source in carbon_source_fit_params:
                # Get fit parameters
                fit_params = carbon_source_fit_params[carbon_source]
                data_range = fit_params['range']
                
                # Select specific data points
                selected_growth_rates = data['growth_rates'][data_range]
                selected_ag_values = data['ag_values'][data_range]
                
                # Calculate 1/growthrate, excluding zero growth rates
                selected_reciprocal = [1/gr if gr > 0 else float('inf') for gr in selected_growth_rates]
                # Remove inf values
                valid_indices = [i for i, r in enumerate(selected_reciprocal) if r != float('inf')]
                selected_reciprocal = [selected_reciprocal[i] for i in valid_indices]
                selected_D_values = [selected_ag_values[i] / 1000 for i in valid_indices]
                
                # Plot pre-overflow data points
                ax.plot(selected_reciprocal, selected_D_values, 'o', mfc='none',markersize = 8,
                        label=display_name, color='red')
            else:
                # If no fit parameters, plot all data points
                ax.plot(reciprocal_growth_rates, D_values, 'o', mfc='none',markersize = 8,
                        label=display_name, color='red')
            
            # If carbon source needs fitting, add linear fit line
            if carbon_source in carbon_source_fit_params:
                # Get fit parameters
                fit_params = carbon_source_fit_params[carbon_source]
                data_range = fit_params['range']
                
                # Select specific data points
                selected_growth_rates = data['growth_rates'][data_range]
                selected_ag_values = data['ag_values'][data_range]
                
                # Calculate 1/growthrate, excluding zero growth rates
                selected_reciprocal = [1/gr if gr > 0 else float('inf') for gr in selected_growth_rates]
                # Remove inf values
                valid_indices = [i for i, r in enumerate(selected_reciprocal) if r != float('inf')]
                selected_reciprocal = [selected_reciprocal[i] for i in valid_indices]
                selected_ag_values = [selected_ag_values[i] for i in valid_indices]
                
                if selected_reciprocal and selected_ag_values:
                    # Convert ag_values to D by dividing by 1000
                    selected_D_values = [ag / 1000 for ag in selected_ag_values]
                    
                    # Perform linear fit
                    slope, intercept, r_squared = linear_fit(selected_reciprocal, selected_D_values)
                    
                    if slope is not None:
                        # Plot fit line
                        x_fit = np.linspace(min(selected_reciprocal), max(selected_reciprocal), 100)
                        y_fit = slope * x_fit + intercept
                        ax.plot(x_fit, y_fit, '--', linewidth=2, 
                                label=f'{display_name} 拟合线', 
                                color='blue')
                        
                        # 在图表中显示拟合方程
                        ax.text(
                            0.5, 0.7,
                            f'y = {slope:.2f}x + {intercept:.2f}\n' + r'$R^2$' + f' = {r_squared:.4f}',
                            transform=ax.transAxes,
                            fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                        )
            
            # 设置标题字体
            ax.set_title(f'{display_name}(respiration)', fontsize=10)
            # 增大横纵坐标标签字体并加粗
            ax.set_xlabel('$1/\mu$ (h)', fontsize=10)
            ax.set_ylabel('D/$\mu$ (kJ/cmol)', fontsize=10)
            # 增大坐标轴刻度字体并加粗
            ax.tick_params(axis='both', which='major', labelsize=10)
            plt.tight_layout()

            
            # Save chart
            save_path = os.path.join(save_folder, f'D_vs_reciprocal_growth_rate_{carbon_source.replace(" ", "_").replace("(", "").replace(")", "")}.pdf')
            plt.savefig(save_path)
            print(f"{carbon_source} 1/Growth Rate chart saved to: {save_path}")
            
            # 关闭图表
            plt.close()
    
    # 创建一个新的集合图，将不同碳源的ag和1/生长率的图表作为子图
    print("\n=== 创建不同碳源ag和1/生长率的集合图 ===")
    # 计算需要的子图数量
    num_carbon_sources = len(all_data)
    # 设置子图布局，每行2个
    rows = (num_carbon_sources + 1) // 2
    cols = 2
    
    # 创建大图，增大画幅
    fig = plt.figure(figsize=(20, 7 * rows))
    
    # 用于标注子图的字母
    subplot_labels = [chr(ord('a') + i) for i in range(num_carbon_sources)]
    
    # 循环为每个碳源创建子图
    for i, (carbon_source, data) in enumerate(all_data.items()):
        if data['growth_rates'] and data['ag_values']:
            # 创建子图
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # 计算1/growthrate，排除growthrate为0的情况
            reciprocal_growth_rates = [1/gr if gr > 0 else float('inf') for gr in data['growth_rates']]
            # 将ag_values除以1000
            ag_values_scaled = [ag / 1000 for ag in data['ag_values']]
            
            # 处理碳源名称，将"Glucose (Thermo)"替换为"Glucose"
            display_name = carbon_source.replace("Glucose (Thermo)", "Glucose")
            
            # 绘制只显示overflow前的数据点
            if carbon_source in carbon_source_fit_params:
                # 获取拟合参数
                fit_params = carbon_source_fit_params[carbon_source]
                data_range = fit_params['range']
                
                # 选择特定数据点
                selected_growth_rates = data['growth_rates'][data_range]
                selected_ag_values = data['ag_values'][data_range]
                
                # 计算1/growthrate，排除growthrate为0的情况
                selected_reciprocal = [1/gr if gr > 0 else float('inf') for gr in selected_growth_rates]
                # 移除inf值
                valid_indices = [i for i, r in enumerate(selected_reciprocal) if r != float('inf')]
                selected_reciprocal = [selected_reciprocal[i] for i in valid_indices]
                selected_ag_scaled = [selected_ag_values[i] / 1000 for i in valid_indices]
                
                # 绘制overflow前的数据点
                ax.plot(selected_reciprocal, selected_ag_scaled, 'o', 
                        label=display_name, color=colors.get(carbon_source, 'black'))
            else:
                # 如果没有拟合参数，绘制所有数据点
                ax.plot(reciprocal_growth_rates, ag_values_scaled, 'o', 
                        label=display_name, color=colors.get(carbon_source, 'black'))
            
            # 如果是需要拟合的碳源，添加线性拟合线
            if carbon_source in carbon_source_fit_params:
                # 获取拟合参数
                fit_params = carbon_source_fit_params[carbon_source]
                data_range = fit_params['range']
                
                # 选择特定数据点
                selected_growth_rates = data['growth_rates'][data_range]
                selected_ag_values = data['ag_values'][data_range]
                
                # 计算1/growthrate，排除growthrate为0的情况
                selected_reciprocal = [1/gr if gr > 0 else float('inf') for gr in selected_growth_rates]
                # 移除inf值
                valid_indices = [i for i, r in enumerate(selected_reciprocal) if r != float('inf')]
                selected_reciprocal = [selected_reciprocal[i] for i in valid_indices]
                selected_ag_values = [selected_ag_values[i] for i in valid_indices]
                
                if selected_reciprocal and selected_ag_values:
                    # 将ag_values除以1000
                    selected_ag_scaled = [ag / 1000 for ag in selected_ag_values]
                    
                    # 执行线性拟合
                    slope, intercept, r_squared = linear_fit(selected_reciprocal, selected_ag_scaled)
                    
                    if slope is not None:
                        # 绘制拟合直线
                        x_fit = np.linspace(min(selected_reciprocal), max(selected_reciprocal), 100)
                        y_fit = slope * x_fit + intercept
                        ax.plot(x_fit, y_fit, '--', linewidth=2, 
                                label=f'{display_name} 拟合线', 
                                color=colors.get(carbon_source, 'black'))
                        
                        # 在图表中显示拟合方程，增大字体并加粗
                        ax.text(0.3, 0.8, f'Fit Equation: $y = {slope:.2f}x + {intercept:.2f}$\n$r^{2} = {r_squared:.4f}$',
                                transform=ax.transAxes, 
                                fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 在子图左上角标注字母，增大字体
            ax.text(0.2, 0.95, f'({subplot_labels[i]})',
                    transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 增大标题字体并加粗
            ax.set_title(f'{display_name}', fontsize=10, fontweight='bold')
            # 设置横纵坐标标签字体
            ax.set_xlabel('$1/\mu$ (h)', fontsize=10, fontweight='bold')
            ax.set_ylabel('D (kJ/cmol)', fontsize=10, fontweight='bold')
            # 设置坐标轴刻度字体
            ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=8)
            ax.tick_params(axis='both', which='minor', width=1.5, length=4)
            ax.grid(True, linestyle='--', alpha=0.7)
            # 设置图例字体
            ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存集合图
    save_path = os.path.join(save_folder, 'all_carbon_sources_ag_vs_reciprocal_growth_rate_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"不同碳源ag和1/生长率的集合图已保存到: {save_path}")
    
    # 关闭图表
    plt.close()
    
    # 定义要绘制的关系类型：'max_growth' 或 'reciprocal_miu1'
    relationship_type = 'reciprocal_miu1'  # 可以根据需要切换为 'max_growth'
    
    # 根据关系类型设置相关参数
    if relationship_type == 'max_growth':
        print("\n\n=== 绘制ag1随max_growth变化图 (含线性拟合) ===")
        ag1_analysis_folder = os.path.join(save_folder, 'ag1_analysis_max_growth')
        x_label = 'max_growth (1/h)'
        title_suffix = 'max_growth'
    else:  # reciprocal_miu1
        print("\n\n=== 绘制ag1随1/miu1变化图 (含线性拟合) ===")
        ag1_analysis_folder = os.path.join(save_folder, 'ag1_analysis_reciprocal_miu1')
        x_label = '$1/\mu_1$ (h)'
        title_suffix = '1_miu1'
    
    # 创建保存ag1分析图表的子文件夹
    if not os.path.exists(ag1_analysis_folder):
        os.makedirs(ag1_analysis_folder)
    
    # 绘制所有碳源汇总的ag1关系图
    plt.figure(figsize=(6, 4))
    
    for carbon_source, data in all_data.items():
        if data['ag1_values'] and data['growth_rates'] and data['miu1_values']:
            # 筛选出有效数据点
            if relationship_type == 'max_growth':
                valid_data = [(ag1, max_growth) for ag1, max_growth, miu1 in zip(data['ag1_values'], data['growth_rates'], data['miu1_values']) 
                             if ag1 is not None and max_growth > 0]
            else:  # reciprocal_miu1
                valid_data = [(ag1, miu1) for ag1, miu1 in zip(data['ag1_values'], data['miu1_values']) 
                             if ag1 is not None and miu1 is not None and miu1 > 0]
            
            if valid_data:
                ag1_values, x_values = zip(*valid_data)
                
                # 计算x轴数据
                if relationship_type == 'max_growth':
                    processed_x_values = x_values
                else:  # reciprocal_miu1
                    processed_x_values = [1/miu1 for miu1 in x_values]
                
                # 将ag1_values除以1000转换为kJ/cmol
                ag1_values_scaled = [ag1 / 1000 for ag1 in ag1_values]
                
                # 处理碳源名称，将"Glucose (Thermo)"替换为"Glucose"
                display_name = carbon_source.replace("Glucose (Thermo)", "Glucose")
                
                # 绘制数据点
                plt.plot(processed_x_values, ag1_values_scaled, 'o', 
                        label=display_name, color=colors.get(carbon_source, 'black'))
    

    plt.xlabel(x_label)
    plt.ylabel('$(D-\mu_{2}a_{G2})/\mu_1$ (kJ/cmol)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(ag1_analysis_folder, f'ag1_vs_{title_suffix}_all.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"汇总图表已保存到: {save_path}")
    
    # 关闭图表
    plt.close()
    
    # 打印拟合结果标题
    print(f"\n=== ag1 vs {title_suffix} 线性拟合结果 ===")
    
    # 分别绘制每个碳源的ag1关系图（含线性拟合）
    for carbon_source, data in all_data.items():
        if data['ag1_values'] and data['growth_rates'] and data['miu1_values']:
            # 筛选出有效数据点
            if relationship_type == 'max_growth':
                valid_data = [(ag1, max_growth) for ag1, max_growth, miu1 in zip(data['ag1_values'], data['growth_rates'], data['miu1_values']) 
                             if ag1 is not None and max_growth > 0]
            else:  # reciprocal_miu1
                valid_data = [(ag1, miu1) for ag1, miu1 in zip(data['ag1_values'], data['miu1_values']) 
                             if ag1 is not None and miu1 is not None and miu1 > 0]
            
            if valid_data:
                ag1_values, x_values = zip(*valid_data)
                
                # 计算x轴数据
                if relationship_type == 'max_growth':
                    processed_x_values = x_values
                else:  # reciprocal_miu1
                    processed_x_values = [1/miu1 for miu1 in x_values]
                
                # 将ag1_values除以1000转换为kJ/cmol
                ag1_values_scaled = [ag1 / 1000 for ag1 in ag1_values]
                
                # 创建图表
                fig = plt.figure(figsize=(3, 2))
                ax = fig.add_subplot(111)
                
                # 处理碳源名称，将"Glucose (Thermo)"替换为"Glucose"
                display_name = carbon_source.replace("Glucose (Thermo)", "Glucose")
                
                # 绘制数据点
                ax.plot(processed_x_values, ag1_values_scaled, 'o',  mfc='none',markersize = 8,
                        label=display_name, color='red')
                
                # 执行线性拟合
                slope, intercept, r_squared = linear_fit(processed_x_values, ag1_values_scaled)
                
                if slope is not None:
                    # 绘制拟合直线
                    x_fit = np.linspace(min(processed_x_values), max(processed_x_values), 100)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, '--', linewidth=2, 
                            label='拟合线', 
                            color='blue')
                    
                    # 在图表中显示拟合方程
                    ax.text(
                        0.53, 0.7,
                        f'y = {slope:.2f}x + {intercept:.2f}\n' + r'$R^2$' + f' = {r_squared:.4f}',
                        transform=ax.transAxes,
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
                    
                    # 打印拟合结果
                    print(f"\n{carbon_source}:")
                    print(f"  斜率: {slope:.6f} kJ/cmol·h")
                    print(f"  截距: {intercept:.6f} kJ/cmol")
                    print(f"  R²值: {r_squared:.6f}")
                
                # 设置标题字体
                ax.set_title(f'{display_name}(overflow) ', fontsize=10)
                # 设置横纵坐标标签字体
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('$(D-\mu_{2}a_{G2})/\mu_1$ (kJ/cmol)', fontsize=10)
                # 设置坐标轴刻度字体
                ax.tick_params(axis='both', which='major', labelsize=10)


                
                # 保存图表
                plt.tight_layout()
                print(fig.get_size_inches())
                save_path = os.path.join(ag1_analysis_folder, 
                                       f'ag1_vs_{title_suffix}_{carbon_source.replace(" ", "_").replace("(", "").replace(")", "")}.pdf')
                plt.savefig(save_path)
                print(f"  图表已保存到: {save_path}")
                
                # 关闭图表
                plt.close()
    
    # 创建一个新的集合图，将不同碳源的ag1和1/miu1的图表作为子图
    print("\n=== 创建不同碳源ag1和1/miu1的集合图 ===")
    # 计算需要的子图数量
    num_carbon_sources = len(all_data)
    # 设置子图布局，每行2个
    rows = (num_carbon_sources + 1) // 2
    cols = 2
    
    # 创建大图，增大画幅
    fig = plt.figure(figsize=(20, 7 * rows))
    
    # 用于标注子图的字母
    subplot_labels = [chr(ord('a') + i) for i in range(num_carbon_sources)]
    
    # 循环为每个碳源创建子图
    for i, (carbon_source, data) in enumerate(all_data.items()):
        if data['ag1_values'] and data['growth_rates'] and data['miu1_values']:
            # 创建子图
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # 筛选出有效数据点
            if relationship_type == 'max_growth':
                valid_data = [(ag1, max_growth) for ag1, max_growth, miu1 in zip(data['ag1_values'], data['growth_rates'], data['miu1_values']) 
                             if ag1 is not None and max_growth > 0]
            else:  # reciprocal_miu1
                valid_data = [(ag1, miu1) for ag1, miu1 in zip(data['ag1_values'], data['miu1_values']) 
                             if ag1 is not None and miu1 is not None and miu1 > 0]
            
            if valid_data:
                ag1_values, x_values = zip(*valid_data)
                
                # 计算x轴数据
                if relationship_type == 'max_growth':
                    processed_x_values = x_values
                else:  # reciprocal_miu1
                    processed_x_values = [1/miu1 for miu1 in x_values]
                
                # 将ag1_values除以1000转换为kJ/cmol
                ag1_values_scaled = [ag1 / 1000 for ag1 in ag1_values]
                
                # 处理碳源名称，将"Glucose (Thermo)"替换为"Glucose"
                display_name = carbon_source.replace("Glucose (Thermo)", "Glucose")
                
                # 绘制数据点
                ax.plot(processed_x_values, ag1_values_scaled, 'o', 
                        label=display_name, color=colors.get(carbon_source, 'black'))
                
                # 执行线性拟合
                slope, intercept, r_squared = linear_fit(processed_x_values, ag1_values_scaled)
                
                if slope is not None:
                    # 绘制拟合直线
                    x_fit = np.linspace(min(processed_x_values), max(processed_x_values), 100)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, '--', linewidth=2, 
                            label=f'{display_name} 拟合线', 
                            color=colors.get(carbon_source, 'black'))
                    
                    # 在图表中显示拟合方程
                    ax.text(0.3, 0.85, f'Fit Equation: $y = {slope:.2f}x + {intercept:.2f}$\n$r^{2}$ = {r_squared:.4f}',
                            transform=ax.transAxes, 
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 在子图左上角标注字母
            ax.text(0.2, 0.95, f'({subplot_labels[i]})',
                    transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 设置标题字体
            ax.set_title(f'{display_name}', fontsize=10, fontweight='bold')
            # 设置横纵坐标标签字体
            ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
            ax.set_ylabel('$(D-\mu_{2}a_{G2})/\mu_1$ (kJ/cmol)', fontsize=10, fontweight='bold')
            # 设置坐标轴刻度字体
            ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=8)
            ax.tick_params(axis='both', which='minor', width=1.5, length=4)
            ax.grid(True, linestyle='--', alpha=0.7)
            # 设置图例字体
            ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存集合图
    save_path = os.path.join(ag1_analysis_folder, f'all_carbon_sources_ag1_vs_{title_suffix}_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"不同碳源ag1和1/miu1的集合图已保存到: {save_path}")
    
    # 关闭图表
    plt.close()
    
    print(f"\n所有图表已保存到文件夹: {save_folder}")

    # 保存所有计算得到的数据到pkl文件，方便后续直接画图
    data_save_path = 'calculated_ag_data.pkl'
    
    # 添加碳源列表到数据字典中，方便加载时使用
    all_data_with_sources = {
        'carbon_sources': list(all_data.keys()),
        **all_data
    }
    
    with open(data_save_path, 'wb') as f:
        pickle.dump(all_data_with_sources, f)
    print(f"\n所有计算数据已保存到: {data_save_path}")
    print(f"保存的碳源数量: {len(all_data_with_sources['carbon_sources'])}")
    print(f"保存的碳源: {all_data_with_sources['carbon_sources']}")
    print("后续可直接加载此文件画图，无需重新计算")

# 加载计算得到的ag数据
def load_calculated_ag_data(file_path='calculated_ag_data.pkl'):
    """
    加载计算得到的ag数据文件
    
    参数:
        file_path: 数据文件路径，默认为'calculated_ag_data.pkl'
    
    返回:
        包含计算数据的字典，键包括:
        - 'carbon_sources': 碳源列表
        - 'growth_rates': 生长率列表
        - 'ag_values': ag值列表
        - 'ag1_values': ag1值列表
        - 'ag2_values': ag2值列表
        - 'miu1_values': miu1值列表
        - 'miu2_values': miu2值列表
    """
    if not os.path.exists(file_path):
        print(f"数据文件 {file_path} 不存在，请先运行计算程序生成数据。")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载数据文件 {file_path}")
        print(f"包含的碳源: {data.get('carbon_sources', [])}")
        print(f"数据项: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        return None

if __name__ == "__main__":
    main()