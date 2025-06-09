class FaceProcessApp {
    constructor() {
        this.defaultConfig = {
            srcDir: './src/demo',
            outputDir: './output/demo',
            threshold: 0.8,
            outputSize: '1024x1024',
            outputFormat: 'jpg',
            types: ['face', 'upper_body', 'half_body'],
            center: false,
            blank: 'keep-blank'
        };
        
        this.currentImages = [];
        this.disabledImages = new Set();
        this.currentTask = null;
        this.websocket = null;
        
        this.init();
    }
    
    init() {
        this.loadConfig();
        this.bindEvents();
        this.refreshImages();
    }
    
    bindEvents() {
        // 配置变更监听
        document.getElementById('srcDir').addEventListener('change', () => {
            this.markConfigChanged('srcDir');
            this.refreshImages();
        });
        
        ['outputDir', 'threshold', 'outputSize', 'outputFormat', 'centerOption', 'blankOption'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                this.markConfigChanged(id);
            });
        });
        
        ['typeFace', 'typeUpperBody', 'typeHalfBody'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                this.markConfigChanged(id);
            });
        });
        
        // 按钮事件
        document.getElementById('saveConfig').addEventListener('click', () => this.saveConfig());
        document.getElementById('resetConfig').addEventListener('click', () => this.resetConfig());
        document.getElementById('refreshImages').addEventListener('click', () => this.refreshImages());
        document.getElementById('toggleView').addEventListener('click', () => this.toggleView());
        document.getElementById('startBatch').addEventListener('click', () => this.startBatchProcess());

        // 排序变更
        document.getElementById('sortOption').addEventListener('change', () => this.sortImages());
    }
    
    loadConfig() {
        const saved = localStorage.getItem('faceProcessConfig');
        const config = saved ? JSON.parse(saved) : this.defaultConfig;
        
        document.getElementById('srcDir').value = config.srcDir;
        document.getElementById('outputDir').value = config.outputDir;
        document.getElementById('threshold').value = config.threshold;
        document.getElementById('outputSize').value = config.outputSize;
        document.getElementById('outputFormat').value = config.outputFormat;
        document.getElementById('centerOption').value = config.center;
        document.getElementById('blankOption').value = config.blank;
        
        // 设置类型复选框
        document.getElementById('typeFace').checked = config.types.includes('face');
        document.getElementById('typeUpperBody').checked = config.types.includes('upper_body');
        document.getElementById('typeHalfBody').checked = config.types.includes('half_body');
    }
    
    getCurrentConfig() {
        const types = [];
        if (document.getElementById('typeFace').checked) types.push('face');
        if (document.getElementById('typeUpperBody').checked) types.push('upper_body');
        if (document.getElementById('typeHalfBody').checked) types.push('half_body');
        
        return {
            srcDir: document.getElementById('srcDir').value,
            outputDir: document.getElementById('outputDir').value,
            threshold: parseFloat(document.getElementById('threshold').value),
            outputSize: document.getElementById('outputSize').value,
            outputFormat: document.getElementById('outputFormat').value,
            types: types,
            center: document.getElementById('centerOption').value === 'true',
            blank: document.getElementById('blankOption').value
        };
    }
    
    markConfigChanged(elementId) {
        const element = document.getElementById(elementId);
        const current = this.getCurrentConfig();
        const isChanged = JSON.stringify(current) !== JSON.stringify(this.defaultConfig);
        
        if (isChanged) {
            element.classList.add('changed-option');
        } else {
            element.classList.remove('changed-option');
        }
    }
    
    async saveConfig() {
        const config = this.getCurrentConfig();
        const name = prompt('请输入配置名称:', 'custom_config');
        
        if (!name) return;
        
        try {
            const response = await fetch('/save_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, ...config })
            });
            
            const result = await response.json();
            if (result.success) {
                localStorage.setItem('faceProcessConfig', JSON.stringify(config));
                alert('配置保存成功!');
            } else {
                alert('保存失败: ' + result.message);
            }
        } catch (error) {
            alert('保存失败: ' + error.message);
        }
    }
    
    resetConfig() {
        this.loadConfig();
        document.querySelectorAll('.changed-option').forEach(el => {
            el.classList.remove('changed-option');
        });
        localStorage.removeItem('faceProcessConfig');
    }
    
    async refreshImages() {
        const srcDir = document.getElementById('srcDir').value;
        if (!srcDir) return;
        
        try {
            const response = await fetch(`/list_images/${encodeURIComponent(srcDir)}`);
            const result = await response.json();
            
            if (result.success) {
                this.currentImages = result.data;
                this.renderImages();
                document.getElementById('startBatch').disabled = this.currentImages.length === 0;
            } else {
                this.showError('获取图片列表失败: ' + result.message);
            }
        } catch (error) {
            this.showError('获取图片列表失败: ' + error.message);
        }
    }
    
    renderImages() {
        const container = document.getElementById('imageContainer');
        
        if (this.currentImages.length === 0) {
            container.innerHTML = '<div class="col-12 text-center text-muted"><p>未找到图片文件</p></div>';
            return;
        }
        
        const sortedImages = this.getSortedImages();
        
        container.innerHTML = sortedImages.map((img, index) => {
            const isDisabled = this.disabledImages.has(img.name);
            // 使用后端图片服务API而不是直接的文件路径
            const imageUrl = `/serve_image/${encodeURIComponent(img.path)}`;
            return `
                <div class="col-md-4 col-lg-3">
                    <div class="image-item ${isDisabled ? 'image-disabled' : ''}" data-name="${img.name}">
                        <img src="${imageUrl}" class="img-fluid rounded" alt="${img.name}">
                        <div class="image-controls">
                            <button class="btn btn-sm ${isDisabled ? 'btn-success' : 'btn-danger'}" 
                                    onclick="app.toggleImage('${img.name}')">
                                <i class="bi ${isDisabled ? 'bi-arrow-clockwise' : 'bi-trash'}"></i>
                            </button>
                        </div>
                        <div class="mt-1">
                            <small class="text-muted">${img.name}</small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    getSortedImages() {
        const sortOption = document.getElementById('sortOption').value;
        const images = [...this.currentImages];
        
        switch (sortOption) {
            case 'name-asc':
                return images.sort((a, b) => a.name.localeCompare(b.name));
            case 'name-desc':
                return images.sort((a, b) => b.name.localeCompare(a.name));
            case 'time-asc':
                return images.sort((a, b) => a.modified - b.modified);
            case 'time-desc':
                return images.sort((a, b) => b.modified - a.modified);
            default:
                return images;
        }
    }
    
    sortImages() {
        this.renderImages();
    }
    
    toggleImage(imageName) {
        if (this.disabledImages.has(imageName)) {
            this.disabledImages.delete(imageName);
        } else {
            this.disabledImages.add(imageName);
        }
        this.renderImages();
    }
    
    toggleView() {
        // 简单的视图切换实现
        const container = document.getElementById('imageContainer');
        const isGrid = container.classList.contains('row');
        
        if (isGrid) {
            container.classList.remove('row', 'g-2');
            container.classList.add('list-view');
        } else {
            container.classList.remove('list-view');
            container.classList.add('row', 'g-2');
        }
    }
    
    async startBatchProcess() {
        const config = this.getCurrentConfig();
        
        if (config.types.length === 0) {
            alert('请至少选择一种输出类型!');
            return;
        }
        
        try {
            // 创建任务
            const response = await fetch('/create_task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    src_dir: config.srcDir,
                    output_dir: config.outputDir,
                    threshold: config.threshold,
                    size: config.outputSize,
                    format: config.outputFormat,
                    types: config.types,
                    center: config.center,
                    blank: config.blank
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentTask = result.data.task_id;
                this.connectWebSocket(this.currentTask);
                this.showProgress();
            } else {
                alert('创建任务失败: ' + result.message);
            }
        } catch (error) {
            alert('创建任务失败: ' + error.message);
        }
    }
    
    connectWebSocket(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/progress/${taskId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.hideProgress();
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket connection closed');
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress':
            case 'status':
                this.updateProgress(data.progress, data.message);
                if (data.status === 'completed' || data.status === 'failed') {
                    setTimeout(() => {
                        this.hideProgress();
                    }, 2000);
                }
                break;
            case 'error':
                alert('任务错误: ' + data.message);
                this.hideProgress();
                break;
        }
    }
    
    showProgress() {
        document.querySelector('.progress-container').style.display = 'block';
        document.getElementById('startBatch').disabled = true;
    }
    
    hideProgress() {
        document.querySelector('.progress-container').style.display = 'none';
        document.getElementById('startBatch').disabled = false;
    }
    
    updateProgress(progress, message) {
        const progressBar = document.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');
        
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';
        progressText.textContent = message;
    }
    
    showError(message) {
        alert(message);
    }
}

// 初始化应用
const app = new FaceProcessApp();