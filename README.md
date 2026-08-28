# 大模型技术地图（GitHub Pages 版）

面向大模型方向的全栈知识体系清单，发布为静态站点：

**https://luxiangz.github.io/llm-tech-map/**

## 更新进度

1. 编辑 `index.md`，把对应条目 `- [ ]` 改为 `- [x]` 即打勾
2. 提交并推送，约 1 分钟后网站自动更新

## 仓库结构

| 文件 | 作用 |
|---|---|
| `index.md` | 技术地图主页（日常只需编辑这一个文件） |
| `notes/` | 知识点的详细笔记页（与地图条目互相链接） |
| `_config.yml` | Jekyll 配置：GFM 引擎 + cayman 主题 |
| `assets/css/style.scss` | 字体与排版覆盖（霞鹜文楷 / 微软雅黑标题） |

## 发布方式

GitHub Pages 从 `main` 分支根目录直接构建（Settings → Pages → Deploy from a branch → main / root）。无需本地构建。
