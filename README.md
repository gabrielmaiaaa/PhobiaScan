# PhobiaScan

Reconhecimento de emoções em tempo real usando **YOLO**, **OpenCV** e **Deep Learning**.  
Projeto desenvolvido para a disciplina de Visão Computacional.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-brightgreen?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)]()
[![Stars](https://img.shields.io/github/stars/gabrielmaiaaa/PhobiaScan?style=social)](https://github.com/gabrielmaiaaa/PhobiaScan)

---

## 🔧 Como rodar o projeto

### 1. Clone o repositório
```bash
git clone https://github.com/gabrielmaiaaa/PhobiaScan
cd PhobiaScan
```

### 2. Crie e ative um ambiente virtual

No **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

No **Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install --no-cache-dir -r requirements.txt
```

### 4. Execute o script principal
```bash
python -m src.cam.yolo_cam
```

---

---

## 🚧 Em breve
- Comparação com outros modelos (DeepFace, MediaPipe...)
- Dataset ajustado para emoções específicas
- Interface de visualização amigável
- Melhorias na detecção em tempo real

---
