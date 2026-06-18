# 🎭 VisioFace Web — Face.ID Premium

Sistema de **reconhecimento facial em tempo real** com interface web, construído em Python com Flask, OpenCV e a biblioteca `face_recognition`.

A aplicação liga a câmera, detecta os rostos ao vivo, compara com uma base de pessoas cadastradas e exibe na tela quem foi identificado — tudo acessível pelo navegador.

---

## ✨ Funcionalidades

- 📹 **Vídeo ao vivo** no navegador (stream MJPEG) com overlay desenhado sobre cada rosto
- 👤 **Cadastro guiado de pessoas** com coleta de múltiplas amostras (5 a 20) em ângulos diferentes
- 🔍 **Reconhecimento em tempo real** com indicação visual por cores:
  - 🟢 Verde — identificado com confiança
  - 🟡 Amarelo — em análise / incerto
  - 🔴 Vermelho — desconhecido ou erro
- 👥 **Múltiplos rostos** detectados e classificados simultaneamente
- 🎯 **Zona de foco central** para desempatar rostos parecidos (ambiguidade)
- 🕓 **Histórico de detecções** com deduplicação e intervalo mínimo entre avistamentos
- 🖼️ **Fotos de referência** das pessoas cadastradas armazenadas em SQLite
- ⚙️ **Configurações ajustáveis** via interface (limiar de confiança, nº de amostras, fotos de referência)

---

## 🧰 Pré-requisitos

- **Python 3.9+**
- Uma **câmera** (webcam) conectada ao servidor
- **Windows** ou **Linux**

> ℹ️ A biblioteca `face_recognition` depende do `dlib`. Este projeto usa `dlib-bin` (binário pré-compilado) para evitar a necessidade de compilar do zero.

---

## 🚀 Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/alissonpk18/visioface-web.git
cd visioface-web

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 🔐 Segurança e LGPD

Este sistema processa **dados biométricos** (dados pessoais sensíveis) conforme definido pela [Lei nº 13.709/2018 (LGPD)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm), Art. 5º, II.

### Configurar senha de acesso (obrigatório em produção)

Defina a variável de ambiente `APP_PASSWORD` antes de iniciar o servidor:

```bash
# Windows
set APP_PASSWORD=sua_senha_aqui
python app.py

# Linux/macOS
export APP_PASSWORD=sua_senha_aqui
python app.py
```

Sem `APP_PASSWORD`, o sistema inicia sem proteção e exibe um aviso no terminal. **Nunca use sem senha em redes compartilhadas ou produção.**

### Retenção de dados

O histórico de avistamentos é purgado automaticamente ao iniciar. O padrão é **90 dias**. Para alterar:

```bash
export HISTORY_RETENTION_DAYS=30   # manter apenas 30 dias
```

### Direitos do titular (Art. 18 da LGPD)

| Direito | Como exercer |
|---------|-------------|
| Exclusão dos dados | Página **Gerenciar Cadastros** → botão Excluir |
| Limpar histórico | Tela principal → botão **Apagar Histórico de Vistos** |
| Limpar todos os cadastros | Gerenciar Cadastros → **Limpar Base** |

---

## ▶️ Como usar

```bash
python app.py
```

Abra o navegador em **http://localhost:5000**

### Fluxo básico

1. **Cadastrar uma pessoa**
   - Acesse a página de cadastros (`/cadastros`)
   - Digite o nome e inicie a coleta
   - Fique **sozinho na câmera**, com o rosto centralizado no quadrado verde
   - Mude levemente o ângulo da cabeça a cada amostra até completar a coleta

2. **Reconhecer**
   - Na tela principal, qualquer rosto cadastrado é identificado automaticamente ao vivo
   - O nome e o nível de confiança aparecem sobre o rosto

3. **Gerenciar**
   - Veja a lista de cadastros, remova pessoas e limpe o histórico pela interface

---

## 📂 Estrutura do projeto

```
visioface-web/
├── app.py                       # Aplicação principal (Flask + lógica de reconhecimento)
├── pkg_resources.py             # Compatibilidade para face_recognition_models
├── requirements.txt             # Dependências Python
├── app.spec                     # Configuração PyInstaller (build do executável)
├── README.md                    # Este arquivo
├── .gitignore
├── faces_db/                    # Banco de dados local (gerado/atualizado pelo app)
│   ├── encodings.npy            # Vetores faciais (128 dimensões por amostra)
│   ├── nomes.json               # Nomes associados a cada encoding
│   ├── references.sqlite3       # Metadados das fotos de referência
│   ├── reference_images/        # Fotos de referência por pessoa
│   └── settings.json            # Configurações do sistema
├── templates/                  # Interface web
│   ├── index.html               # Dashboard principal (vídeo ao vivo)
│   └── cadastros.html           # Gerenciamento de cadastros
└── scripts/
    └── servidor_keep_alive.bat  # Watchdog Windows (reinício automático)
```

---

## 🧩 Arquitetura interna (`app.py`)

O código é organizado em componentes com responsabilidades únicas:

| Componente | Responsabilidade |
|------------|------------------|
| `CameraWorker` | Captura de frames da webcam em thread dedicada |
| `FaceRepository` | Armazena/carrega os encodings faciais e os nomes |
| `ReferenceImageStore` | Guarda as fotos de referência (SQLite + arquivos) |
| `HistoryRepository` | Registra o histórico de avistamentos (JSONL) com deduplicação |
| `SettingsRepository` | Persiste as configurações em `settings.json` |
| `WebTracker` | Orquestra a captura, detecção, classificação e desenho do overlay |
| `Flask` | Servidor web que serve a interface e a API |

**Fluxo resumido:**
`frame da câmera → detecção/encoding → classificação (multi-face, ambiguidade, zona de foco, histórico) → overlay OpenCV + APIs Flask`

---

## 🔌 Rotas da API

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Dashboard principal |
| `/cadastros` | GET | Página de gerenciamento de cadastros |
| `/video_feed` | GET | Stream MJPEG da câmera com overlay |
| `/api/status` | GET | Status atual da detecção (nome, distância, contadores) |
| `/api/settings` | GET / POST | Ler ou atualizar as configurações |
| `/api/history` | GET | Histórico de detecções (param. `limit`, 1–200) |
| `/api/history/clear` | POST | Limpar o histórico |
| `/api/registrations` | GET | Lista de cadastros e contagem de amostras |
| `/api/registration/delete` | POST | Remover um cadastro (`{ "name": "..." }`) |
| `/api/registration/clear` | POST | Remover todos os cadastros |
| `/api/action` | POST | Ações: `register`, `validate`, `cancel` |

---

## ⚙️ Configurações

Ficam em `faces_db/settings.json` e podem ser ajustadas pela interface:

```json
{
  "confirmed_score_threshold": 0.199,
  "samples_needed": 11,
  "max_reference_photos_per_person": 15
}
```

| Configuração | Faixa | Descrição |
|--------------|-------|-----------|
| `confirmed_score_threshold` | 0.05 – 0.99 | Pontuação mínima para confirmar uma identidade. Quanto maior, mais rigoroso |
| `samples_needed` | 2 – 20 | Número de amostras coletadas por pessoa no cadastro |
| `max_reference_photos_per_person` | 1 – 60 | Máximo de fotos de referência guardadas por pessoa |

---

## 🛠️ Gerando o executável (Windows)

```bash
pip install pyinstaller
pyinstaller app.spec
```

O executável é gerado em `dist/app/app.exe`.

Para manter o servidor sempre ativo, use o watchdog que reinicia o app automaticamente caso ele feche:

```bash
scripts\servidor_keep_alive.bat
```

---

## 📝 Notas

- Os encodings faciais são **vetores numéricos de 128 dimensões** — o app não armazena imagens dos rostos para identificação, apenas para referência visual.
- A pasta `faces_db/` contém dados gerados em uso; faça backup dela se quiser preservar os cadastros.
- Arquivos de runtime (`capture_history.jsonl`, `error.log`), caches e artefatos de build são ignorados pelo Git (ver `.gitignore`).
