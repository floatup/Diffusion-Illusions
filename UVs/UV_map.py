import numpy as np
from PIL import Image
from scipy.spatial import Voronoi
from matplotlib.path import Path
import math
import random
import argparse

class UV_Voronoi:
    def __init__(self):
        self.size = 256
        self.cellnum = 20
        self.jitter = 0.08
        self.rotate_range = math.pi
        self.scale_range = (0.6, 1.3)
        self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.file = "voronoi_uv.png"

    def generate(self):
        # 生成 Voronoi 种子点，并基于点生成 Voronoi 图
        points = np.random.rand(self.cellnum, 2)
        Voronoi(points)
        cell_params = []
        for i in range(self.cellnum):
            cell_params.append({
                "angle": random.uniform(-self.rotate_range, self.rotate_range),
                "scale": random.uniform(*self.scale_range),
                "offset": np.random.uniform(-self.jitter, self.jitter, size=2)
            })
        uv_map = np.zeros((self.size, self.size, 3), dtype=np.float32)
        for y in range(self.size):
            for x in range(self.size):
                p = np.array([x / (self.size - 1), y / (self.size - 1)])

                # Voronoi 归属
                d = np.sum((points - p) ** 2, axis=1)
                idx = np.argmin(d)
                site = points[idx]
                prm = cell_params[idx]

                # 局部坐标
                local = p - site
                c, s = math.cos(-prm["angle"]), math.sin(-prm["angle"])
                local = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
                local /= prm["scale"]

                # 映射回 UV
                uv = site + local + prm["offset"]

                # 保证 UV 合法（填补而不是黑洞）
                uv = np.clip(uv, 0.0, 1.0)

                uv_map[y, x, 0] = uv[0]
                uv_map[y, x, 1] = uv[1]
                uv_map[y, x, 2] = 0.0
        img = (uv_map * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(self.file)
        print(f"Saved {self.file}")

class UV_Tangram:
    def __init__(self):
        self.size_x = 384
        self.size_y = 320
        self.file = "tangram_uv.png"
    def generate_tangram_shapes(self):
        """
        定义七巧板的形状轮廓。
        每个形状通过一组顶点定义。
        """
        shapes = [
            # 大三角形1
            [(0, 0), (128, 128), (0, 256)],
            # 大三角形2
            [(128, 128), (0, 256), (256, 256)],
            # 小三角形1
            [(0, 0), (128, 0), (64, 64)],
            # 正方形
            [(128, 0), (64, 64), (128, 128), (192, 64)],
            # 小三角形2
            [(192, 64), (128, 128), (192, 192)],
            # 平行四边形
            [(192, 64), (192, 192), (256, 256), (256, 128)],
            # 中三角形
            [(128, 0), (256, 0), (256, 128)],
        ]
        return shapes
    
    def transfer(self, index, x, y):
        match index:
            case 0:
                xx = -y + 256
                yy = x + 192
            case 1:
                xx = x
                yy = y - 64
            case 2:
                xx = -x + 256
                yy = -y + 64
            case 3:
                xx = x + 128
                yy = y + 192
            case 4:
                xx = -x + 512
                yy = -y + 320
            case 5:
                xx = x + 64
                yy = y
            case 6:
                xx = x
                yy = y + 64
            case _:
                xx = 0
                yy = 0
        return 384 - xx, 320 - yy
    def generate(self):
        uv_map = np.zeros((self.size_y+1, self.size_x+1, 3), dtype=np.float32)
        shapes = self.generate_tangram_shapes()
        for index, shape in enumerate(shapes):
            poly_path = Path(shape)
            min_x = int(min(v[0] for v in shape))
            max_x = int(max(v[0] for v in shape))
            min_y = int(min(v[1] for v in shape))
            max_y = int(max(v[1] for v in shape))
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if poly_path.contains_point((x, y)):
                        xx, yy = self.transfer(index, x, y)
                        uv_map[yy, xx, 0] = x / 256
                        uv_map[yy, xx, 1] = y / 256
                        uv_map[yy, xx, 2] = 0.0
        img = (uv_map * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(self.file)
        print(f"Saved {self.file}")
        pass

def help():
    print("+====================================+")
    print("")
    print("You should run: python UV_map.py --run=[type]")
    print("The 'type' is the type of uv map. Current optional types:")
    print("    - Voronoi")
    print("    - Tangram")
    print("")
    print("+====================================+")

parser = argparse.ArgumentParser(description='The configs')
parser.add_argument('--run', type=str, help='生成的图像类型', default='Voronoi')
# parser.add_argument('--help', type=str, default='h')
args = parser.parse_args()
# 读取参数运行相应函数
if args.run == 'Voronoi':
    UV_Voronoi().generate()
if args.run == 'Tangram':
    UV_Tangram().generate()
else:
    help()
    raise AssertionError("Error UV type!!!!")