# 部署指南 (Deployment Guide)

## 1. 分支策略与环境映射

| 动作 | 触发工作流 | 结果 |
|------|------------|------|
| `feature/*` → `dev` 的 PR | `ci.yml` | ruff + black + pytest + 构建镜像 + 容器冒烟测试；失败则禁止合并 |
| `dev` → `staging` 的 PR | `ci.yml` | 同上（`staging` 分支的 PR 也会跑 CI） |
| 合并到 `staging` | `deploy-staging.yml` | 生成 staging `.env`，构建 `ml-app:staging` |
| 合并到 `main` | `deploy-production.yml` | 生成 production `.env`，构建 `ml-app:production` |

`dev`、`staging`、`main` 均已开启分支保护，PR 合并前必须通过 required status checks（`Lint & Test`、`Build & Smoke-test Image`）。

## 2. GitHub Secrets 配置

本项目使用 GitHub Environments 做环境隔离，需要在对应 environment 下配置：

| Environment | Secret | 用途 |
|-------------|--------|------|
| `staging` | `STAGING_ENV_FILE` | 写入 `.env`，供 staging 镜像使用 |
| `production` | `PRODUCTION_ENV_FILE` | 写入 `.env`，供 production 镜像使用 |

> `.env` 被 `.dockerignore` 排除，不会被打进镜像；它的用途是给部署平台（Compose / K8s）注入运行时变量。

## 3. DVC 数据凭证

CI 不执行 `dvc pull`，因此**不需要**把 Azure 凭证放进 GitHub Secrets。本地训练时：

```bash
# .dvc/config.local（不入库）
# ['remote "myazure"']
#     url = azure://dvc
#     sas_token = sp=...&sig=...
dvc pull
```

SAS token 有效期至 2026-12-12；过期后到 Azure Portal 重新生成，覆盖 `.dvc/config.local` 即可。

## 4. 构建与运行镜像

```bash
# 与 CI 一致
docker build -t ml-app:local .
docker run -d -p 8000:8000 --name ml-app ml-app:local

curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"A wonderful, brilliant film."}'
# {"sentiment":"positive","confidence":0.7952}
```

镜像内模型路径由 `MODEL_PATH` 环境变量控制，默认 `/app/models/model_v1.pkl`；如需换模型，挂载新文件并覆盖该变量即可。

## 5. 健康检查与排障

- `/health` 返回 `{"status":"ok","model_loaded":<bool>}`。
- `model_loaded=false` 说明镜像里缺 `models/model_v1.pkl`：检查该文件是否已提交、`.dockerignore` 是否误排除 `models/`。
- `/predict` 在模型缺失时返回 **503**（而不是 200 + error 字段），便于编排系统识别。
