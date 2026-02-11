# 图像匹配项目

一个基于Python的智能图像匹配系统，支持多种匹配算法，提供Web界面和批量处理功能，当前采用混合哈希算法进行匹配。

## 功能特性

- **多种匹配算法**: 支持传统算法（感知哈希、结构匹配）以及后续拓展深度学习算法（CLIP）
- **Web界面**: 提供友好的Web界面，支持图片上传和实时匹配
- **批量处理**: 支持批量图片匹配和文件夹级别的匹配
- **智能排序**: 基于结构相似度的智能顺序重排和冲突消解
- **结果导出**: 支持将匹配结果打包为ZIP文件下载
- **配置灵活**: 通过YAML配置文件灵活调整算法参数

## 项目结构

```
img_match_project-master/
├── app/                    # UI & 入口
│   ├── app.py            # Flask应用主文件
│   ├── controller.py       # 调度入口
│   └── index.html        # HTML界面
│
├── core/                   # 核心逻辑（稳定区）
│   ├── pipeline.py         # 总流程编排
│   ├── matcher_base.py     # 匹配器接口
│   ├── assignment.py       # 顺序重排和冲突消解
│   └── result.py           # 统一结果结构
│
├── algorithms/             # 可插拔算法实现
│   ├── traditional/
│   │   ├── structure.py    # 结构提取
│   │   ├── phash.py        # 感知哈希
│   │   ├── hash_matcher.py # 哈希匹配器
│   │   └── matcher.py      # 传统 matcher
│   │
│   └── deep/               # 深度学习算法
│       ├── clip_encoder.py # CLIP编码器
│       └── clip_matcher.py # CLIP匹配器
│
├── utils/
│   ├── image_io.py         # 图像读写工具
│   ├── similarity.py       # 相似度计算
│   ├── zip_export.py       # ZIP导出工具
│   └── logger.py           # 日志工具
│
├── config/
│   ├── traditional.yaml    # 传统算法配置
│   └── deep.yaml           # 深度学习算法配置
│
├── debug/                  # 调试输出目录
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明文档
```

## 安装说明

### 环境要求

- Python 3.8+
- pip

### 安装步骤

1. 克隆或下载项目到本地

2. 安装依赖包

```bash
pip install -r requirements.txt
```

依赖包包括：
- Flask >= 2.3.0 (Web框架)
- opencv-python >= 4.8.0 (图像处理)
- Pillow >= 10.0.0 (图像处理)
- numpy >= 1.25.0 (数值计算)
- scipy >= 1.11.0 (科学计算)
- imagehash >= 4.3.1 (哈希算法)
- pyyaml >= 6.0 (配置文件)
- loguru >= 0.7.0 (日志记录)
- tqdm >= 4.66.0 (进度条)

## 使用说明

### 启动Web应用

```bash
python app/app.py
```

应用启动后会自动在浏览器中打开 `http://127.0.0.1:5000`

### Web界面使用

1. **单次匹配模式**
   - 在A组上传第一批图片（原图，有顺序）
   - 在B组上传第二批图片（重绘图，乱序）
   - 点击"开始匹配"按钮
   - 查看匹配结果，可下载匹配后的图片

2. **文件夹匹配模式**
   - 输入A组文件夹路径
   - 输入B组文件夹路径
   - 输入输出文件夹路径
   - 点击"开始匹配"按钮
   - 匹配结果将保存到输出文件夹

### 配置文件说明

#### 传统算法配置 (config/traditional.yaml)

```yaml
# 感知哈希配置
phash:
  hash_size: 8              # 哈希大小
  highfreq_factor: 4       # 高频因子

# 结构提取配置
structure:
  canny_threshold1: 100    # Canny边缘检测低阈值
  canny_threshold2: 200    # Canny边缘检测高阈值
  min_contour_area: 100    # 最小轮廓面积

# 匹配参数
matcher:
  similarity_threshold: 0.8 # 相似度阈值
  max_results: 10          # 最大结果数
```

## 核心算法

### 1. 传统匹配算法

- **感知哈希**: 使用感知哈希算法计算图片的相似度
- **结构匹配**: 基于Canny边缘检测和轮廓提取的结构相似度
- **混合匹配**: 结合感知哈希和结构匹配的综合评分

### 2. 深度学习匹配算法

- **CLIP编码**: 使用OpenAI的CLIP模型提取图像特征
- **语义匹配**: 基于语义特征的图像匹配

### 3. 智能排序

- **顺序重排**: 基于结构相似度重排B组图片顺序
- **冲突消解**: 解决多对一匹配冲突
- **置信度评估**: 评估匹配结果的置信度

## API接口

### POST /api/match

处理图片匹配请求

**请求参数**:
- `imagesA[]`: A组图片文件（多文件上传）
- `imagesB[]`: B组图片文件（多文件上传）

**返回结果**:
```json
{
  "images": ["base64编码的图片数组"],
  "info": "匹配信息",
  "filenames": ["文件名数组"]
}
```

### GET /api/download

下载匹配后的图片ZIP文件

### POST /api/trainset-match

处理文件夹级别的匹配

**请求参数**:
```json
{
  "folderA": "A组文件夹路径",
  "folderB": "B组文件夹路径",
  "outputFolder": "输出文件夹路径"
}
```

## 开发说明

### 添加新的匹配算法

1. 在 `algorithms/` 目录下创建新的算法模块
2. 继承 `BaseMatcher` 类
3. 实现 `prepare()` 和 `match()` 方法
4. 在配置文件中添加相应的配置项

### 扩展Web功能

1. 在 `app/app.py` 中添加新的路由
2. 在 `app/controller.py` 中实现业务逻辑
3. 在 `app/index.html` 中添加前端界面

## 注意事项

1. 支持的图片格式: JPG, PNG, BMP等常见格式
2. 建议图片尺寸不超过 4096x4096
3. 批量处理时注意内存使用情况
4. 匹配精度取决于图片质量和算法参数配置

