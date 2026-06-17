# VisioFace Web

Sistema de reconhecimento facial em tempo real com interface web, construído com Python, Flask e OpenCV.

## Pré-requisitos

- Python 3.9+
- Câmera conectada ao servidor
- Sistema operacional: Windows ou Linux

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd visioface-web

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt
```

## Executando

```bash
python app.py
```

Acesse no navegador: **http://localhost:5000**

## Estrutura do Projeto

```
visioface-web/
├── app.py                   # Aplicação principal (Flask + reconhecimento facial)
├── pkg_resources.py         # Compatibilidade para face_recognition_models
├── requirements.txt         # Dependências Python
├── app.spec                 # Configuração PyInstaller (build de executável)
├── faces_db/                # Banco de dados de rostos
│   ├── encodings.npy        # Encodings faciais (NumPy)
│   ├── nomes.json           # Nomes associados aos encodings
│   ├── references.sqlite3   # Fotos de referência
│   └── settings.json        # Configurações do sistema
├── templates/               # Templates HTML
│   ├── index.html           # Dashboard principal
│   └── cadastros.html       # Gerenciamento de cadastros
└── scripts/
    └── servidor_keep_alive.bat  # Script Windows para reinício automático
```

## Rotas da API

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Dashboard principal |
| `/cadastros` | GET | Página de gerenciamento |
| `/video_feed` | GET | Stream MJPEG da câmera |
| `/api/status` | GET | Status atual da detecção |
| `/api/settings` | GET/POST | Configurações do sistema |
| `/api/history` | GET | Histórico de detecções |
| `/api/history/clear` | POST | Limpar histórico |
| `/api/registrations` | GET | Lista de cadastros |
| `/api/registration/delete` | POST | Remover cadastro |
| `/api/registration/clear` | POST | Limpar todos os cadastros |
| `/api/action` | POST | Ações: `register`, `validate`, `cancel` |

## Gerando o Executável (Windows)

```bash
pip install pyinstaller
pyinstaller app.spec
```

O executável será gerado em `dist/app/app.exe`.  
Use `scripts/servidor_keep_alive.bat` para reinício automático do servidor.

## Configurações

As configurações ficam em `faces_db/settings.json`:

```json
{
  "confirmed_score_threshold": 0.199,
  "samples_needed": 11,
  "max_reference_photos_per_person": 15
}
```
