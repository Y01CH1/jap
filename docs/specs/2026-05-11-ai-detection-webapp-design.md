# AI 图片/视频鉴别工具 — 设计文档

**日期**: 2026-05-11
**状态**: MVP 阶段

## 概述

面向所有用户的 Web 应用，上传图片检测其是否为 AI 生成，返回概率分数和判定结论。MVP 仅支持图片检测，服务端单模型推理，无需登录。

## 架构

```
frontend (React + Vite)          backend (Python FastAPI)
┌──────────────────────┐         ┌─────────────────────────┐
│ UploadZone           │  POST   │ POST /api/v1/detect      │
│ ResultCard           │ ──────> │   → detector.detect()   │
│ HistoryList          │ <────── │   → DetectionResult      │
│ useDetection hook    │  JSON   │ GET /api/v1/health        │
└──────────────────────┘         └─────────────────────────┘
         │                                 │
         └──────── Vite proxy ─────────────┘
              /api/* → localhost:8000
```

前后端完全分离。开发时通过 Vite proxy 转发 `/api/*` 到后端。无数据库，无持久化。

### 目录结构

```
jap/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/detection.py    # API 端点
│   │   ├── services/
│   │   │   ├── detector.py         # 检测模型抽象接口
│   │   │   ├── models/base.py      # 当前服务端模型实现
│   │   │   └── rate_limiter.py     # IP 频率限制
│   │   ├── schemas/detection.py    # Pydantic 模型
│   │   └── core/config.py
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   └── HistoryList.tsx
│   │   ├── services/api.ts
│   │   ├── hooks/useDetection.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
└── docs/superpowers/specs/
```

## 后端设计

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/detect` | 上传图片，返回检测结果 |
| GET | `/api/v1/health` | 健康检查 + 模型加载状态 |

### POST /api/v1/detect

请求：`multipart/form-data`
- `file`：图片文件（jpg/png/webp，最大 16MB）
- `model`：可选，预留字段，当前固定 `default`

响应 200：
```json
{
  "id": "req_abc123",
  "score": 0.87,
  "verdict": "likely_ai",
  "model_used": "default",
  "inference_time_ms": 320,
  "details": {
    "threshold": 0.5
  }
}
```

- `score`：0-1 浮点数，接近 1 表示更可能是 AI 生成
- `verdict`：`likely_ai` (score > 0.7)、`likely_real` (score < 0.3)、`uncertain` (0.3 ≤ score ≤ 0.7)
- `inference_time_ms`：推理耗时

错误响应：
- `413`：文件过大
- `429`：频率限制触发
- `422`：文件格式不支持
- `500`：模型推理失败

### 速率限制

基于客户端 IP 的滑动窗口限制：
- 默认：每 IP 每分钟 20 次请求
- 使用 `fastapi-limiter`，开发阶段用内存后端（生产切 Redis）
- 超限返回 429 + `Retry-After` 头

### 检测模型

```
detector.py (抽象接口)
    └── models/base.py (当前实现: PyTorch 服务端推理)
    └── models/onnx.py (后续实现: ONNX 浏览器端推理)
```

- MVP 使用 HuggingFace 预训练模型（如 DIRE 系列）
- 服务启动时加载模型到内存
- `detector.py` 定义统一接口 `def detect(image) -> DetectionResult`
- 后续加 ONNX 实现只需新增类，API 层和行为不变

## 前端设计

### 页面结构

单页应用，所有功能在一个页面内完成。

```
┌──────────────────────────────┐
│ Header (logo + 标题)         │
├──────────────────────────────┤
│ UploadZone                   │
│ 拖拽或点击上传图片            │
├──────────────────────────────┤
│ ResultCard (检测完成后展示)   │
│ 环形图 + verdict + 耗时      │
├──────────────────────────────┤
│ HistoryList                  │
│ 本次会话检测记录              │
└──────────────────────────────┘
```

### 组件树

- **App** — 根组件，管理路由（当前单页面，后续可扩展）
- **UploadZone** — 拖拽/点击上传，react-dropzone 实现，上传前预览
- **ResultCard** — 检测结果：SVG 环形概率图、verdict 标签、推理耗时
- **HistoryList** — 会话级历史记录（存储于内存/state，刷新后清空）

### 状态流 (useDetection hook)

```
idle → uploading → detecting → done
  │        │           │          │
  └────────┴───────────┴──────────┴── error (任意阶段可触发)
```

- `idle`：初始，显示上传区域
- `uploading`：正在上传文件，显示进度条
- `detecting`：服务端正推理，显示扫描动画
- `done`：展示 ResultCard
- `error`：错误提示 + 重新检测按钮

### 前端依赖

- **react-dropzone**：拖拽上传
- **axios**：HTTP 请求（带上传进度回调）
- **CSS modules**：样式，不引入组件库
- **纯 SVG/CSS**：环形图，不引入图表库

## 关键设计决策

1. **detector 抽象层**：API 和模型实现之间有一层抽象接口，后续切换模型或加 ONNX 前端推理不影响 API 层
2. **无数据库**：MVP 不需要持久化，会话历史存在前端内存，刷新即清
3. **verdict 三档而非二档**：`uncertain` 区间给用户透明感，避免二元判断的误导
4. **前端纯静态**：Vite 构建的纯静态资源，可部署到 CDN，不依赖 Node.js 服务端渲染
5. **前后端通信仅通过 JSON API**：为后续 B/C/D 形态（浏览器扩展、App、API 服务）共用后端打好基础

## 后续扩展预留

- **浏览器端推理**：`detector.py` 抽象层已有接口，加 `models/onnx.py` 实现即可。前端加 `useLocalDetection` hook
- **视频检测**：新增 `POST /api/v1/detect/video`，复用 detector 抽象
- **用户系统**：加 auth router + 中间件，API 设计不变
- **浏览器扩展**：复用同一套 API
- **移动 App**：复用同一套 API
