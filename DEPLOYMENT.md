# 部署指南 (Deployment Guide)

## 分支策略与环境映射
- `feature/*` PR 到 `dev`：触发 CI (test + lint + docker build)，失败则禁止合并。
- `dev` 合并到 `staging`：触发 `Deploy Staging`，构建 staging 镜像。
- `staging` 合并到 `main`：触发 `Deploy Production`，构建 production 镜像。

## GitHub Environments Secrets
本项目使用 GitHub Environments 进行环境隔离，必须在对应环境下配置以下 Secret：
- **staging 环境**: 
  - `STAGING_ENV_FILE`: 用于生成预发布环境的 .env 文件。
- **production 环境**: 
  - `PRODUCTION_ENV_FILE`: 用于生成生产环境的 .env 文件。
- **Repository Secrets** (DVC 等全局使用): 
  - `AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_KEY`