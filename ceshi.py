import os
import numpy as np
import matplotlib.pyplot as plt
# 设置工作目录下的vmatrix文件夹路径

def get_idx(x):
    return 1

vmatrix_path = os.path.join(os.getcwd(), 'bfa')
plt_data = []
name = []
# 检查vmatrix文件夹是否存在
plt.rc('font',family='Times New Roman')

if not os.path.exists(vmatrix_path):
    print("vmatrix文件夹不存在，请检查路径。")
else:
    # 遍历vmatrix文件夹中的所有文件
    for filename in os.listdir(vmatrix_path):
        # 检查文件扩展名是否为.npy
        test_list = ["bfa_standing.npy", "bfa_sitting.npy", "bfa_squatting.npy", "bfa_no_obstacle.npy"]
        if filename != test_list[0]:
            continue
        if filename.endswith('.npy'):
            # 构造完整的文件路径
            file_path = os.path.join(vmatrix_path, filename)
            # 读取.npy文件
            try:
                array_data = np.load(file_path)
                print(array_data.shape)
                # 2000 * 234 * 10
                print(f"读取了文件：{filename}，内容为：")

                # 假设我们直接使用array_data来创建四个子图，这里为了示例，我假设我们要显示array_data的前四个切片
                # 如果你需要展示magnitude_array的不同切片或其他处理，请相应调整
                #subplots_data = array_data.reshape((array_data.shape[0], -1))
                name.append(filename)
                for i in range(4):
                    subplots_data = array_data[1000:1010, :, i]
                    plt_data.append(subplots_data)
            except Exception as e:
                print(f"读取文件{filename}时发生错误：{e}")
    print(len(plt_data))

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 5))
    for idx, ax in enumerate(axes.flatten()):
        reshaped_array = plt_data[idx]
        for i in range(10):
            single_frame = reshaped_array[i, :]
            ax.plot(single_frame)  # 绘制折线图
        ax.set_title(f"angle {idx + 1}", fontsize=16)


     #添加一个全局的颜色条，共享给所有子图
    #fig.colorbar(im, ax=axes.ravel().tolist(), label='Value Intensity')

    # 调整子图间距
    plt.rcParams['font.family'] = 'SongTi SC'  # 替换为你选择的字体
    plt.tight_layout()
    #plt.suptitle('站立', fontsize=16, y = 1)
    #plt.title("Standing" ,loc='center')
    # 显示图像
    plt.show()