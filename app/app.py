# Flask应用主文件
import sys
import os
import base64
import cv2
import numpy as np
import webbrowser
import threading
import time
from io import BytesIO

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, send_file, render_template
from controller import Controller

# 获取模板目录路径（支持打包后的环境）
if getattr(sys, 'frozen', False):
    # 打包后的环境
    template_dir = os.path.join(sys._MEIPASS, 'app')
else:
    # 开发环境
    template_dir = os.path.dirname(os.path.abspath(__file__))

# 创建Flask应用
app = Flask(__name__, template_folder=template_dir)

# 延迟初始化控制器
controller = None

def get_controller():
    """获取控制器实例（懒加载）"""
    global controller
    if controller is None:
        from controller import Controller
        controller = Controller()
    return controller

def open_browser():
    """延迟打开浏览器，等待服务器启动"""
    time.sleep(0.8)
    webbrowser.open('http://127.0.0.1:5000')

@app.route('/')
def index():
    """主页路由，返回HTML页面"""
    return render_template('index.html')

@app.route('/api/match', methods=['POST'])
def match_images():
    """处理图片匹配请求"""
    try:
        # 获取上传的图片
        imagesA = []
        imagesB = []
        
        # 处理A组图片 - 保存为元组 (numpy数组, 文件名)
        for key in request.files:
            if key.startswith('imagesA['):
                file = request.files[key]
                # 读取图片
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    imagesA.append((img, file.filename))
        
        # 处理B组图片 - 保存为元组 (numpy数组, 文件名)
        for key in request.files:
            if key.startswith('imagesB['):
                file = request.files[key]
                # 读取图片
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    imagesB.append((img, file.filename))
        
        if not imagesA or not imagesB:
            return jsonify({'error': '请上传有效的图片'}), 400
        
        # 调用控制器进行匹配
        matched_images, match_info = get_controller().process_batches(imagesA, imagesB)
        
        # 将匹配后的图片转换为base64格式
        base64_images = []
        for img in matched_images:
            # 确保图片是RGB格式，然后转换为BGR格式供cv2.imencode使用
            img_bgr = img
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 如果是RGB格式，转换为BGR格式
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 转换为PNG
            _, buffer = cv2.imencode('.png', img_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(img_base64)
        
        # 返回结果
        return jsonify({
            'images': base64_images,
            'info': match_info,
            'filenames': get_controller().matched_image_filenames
        })
        
    except Exception as e:
        print(f"匹配错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'匹配失败: {str(e)}'}), 500

@app.route('/api/download')
def download_results():
    """下载匹配后的图片ZIP文件"""
    try:
        # 调用控制器生成zip文件
        zip_path = get_controller().export_matched_images()
        
        if not zip_path or not os.path.exists(zip_path):
            return jsonify({'error': '无法生成下载文件，可能尚未执行匹配'}), 400
        
        # 获取文件名和目录
        zip_dir = os.path.dirname(zip_path)
        zip_filename = os.path.basename(zip_path)
        
        # 发送文件，不使用cache_timeout参数
        response = send_file(zip_path, 
                            as_attachment=True, 
                            download_name='matched_images.zip')
        
        # 设置适当的Content-Type
        response.headers['Content-Type'] = 'application/zip'
        # 设置缓存控制头，避免浏览器缓存
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        # 确保在请求完成后删除临时文件
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                # 删除整个临时目录
                if os.path.exists(zip_dir):
                    import shutil
                    shutil.rmtree(zip_dir)
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")
        
        return response
        
    except Exception as e:
        print(f"下载错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/api/trainset-match', methods=['POST'])
def trainset_match():
    """处理trainset模式的图片匹配请求"""
    try:
        # 获取请求数据
        data = request.get_json()
        folderA = data.get('folderA')
        folderB = data.get('folderB')
        outputFolder = data.get('outputFolder')
        
        if not folderA or not folderB or not outputFolder:
            return jsonify({'error': '请提供完整的文件夹路径'}), 400
        
        # 检查文件夹是否存在
        if not os.path.exists(folderA) or not os.path.isdir(folderA):
            return jsonify({'error': f'A组文件夹不存在或不是目录: {folderA}'}), 400
        if not os.path.exists(folderB) or not os.path.isdir(folderB):
            return jsonify({'error': f'B组文件夹不存在或不是目录: {folderB}'}), 400
        
        # 调用控制器进行trainset匹配
        match_info = get_controller().process_trainset(folderA, folderB, outputFolder)
        
        # 返回结果
        return jsonify({
            'info': match_info
        })
        
    except Exception as e:
        print(f"Trainset匹配错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Trainset匹配失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)
