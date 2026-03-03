# RunPythonTool FastAPI 服务说明

## 1. 当前机器核数结论
- `nproc` 可用 vCPU: **64**
- `lscpu` 物理核: **64**（2 路 * 32 核）
- `lscpu` 逻辑核: **128**（超线程）

本服务按 `nproc=64` 作为可用并发上限来定参数，避免盲目按 128 逻辑核超配。

## 2. 已完成的代码改造
文件: `tools_server/RunPythonTool.py`

- 新增 FastAPI 服务接口:
  - `GET /healthz`
  - `GET /stats`
  - `POST /run`
- 新增异步执行: `ToolSandbox.run_async(...)`
- 保留 `ProcessPoolExecutor + spawn` 进程隔离
- 增加并发背压机制:
  - `RUNPY_MAX_INFLIGHT` 控制同时在途请求上限
  - 排队等待超时后直接返回 busy（429），防止堆积导致全体超时
- 增加超时保护:
  - worker 内 `SIGALRM` 墙钟超时
  - 外层超时后自动重建进程池，避免卡死 worker 逐渐耗尽
- 增加 worker 回收:
  - `RUNPY_WORKER_MAX_TASKS` 达到任务数后自动换新进程，降低长期碎片化风险

## 3. 当前已启动参数（已生效）
后台进程 PID: `330357`
端口: `18080`

```bash
env RUNPY_MAX_WORKERS=32 \
    RUNPY_MAX_INFLIGHT=128 \
    RUNPY_TIMEOUT=15 \
    RUNPY_QUEUE_WAIT_TIMEOUT=2 \
    RUNPY_WORKER_MAX_TASKS=200 \
    RUNPY_HOST=0.0.0.0 \
    RUNPY_PORT=18080 \
    python tools_server/RunPythonTool.py
```

参数解释:
- `MAX_WORKERS=32`: 对 64 vCPU 取 50%，给系统、网络、日志等留余量
- `MAX_INFLIGHT=128`: 允许一定排队，但避免无限堆积
- `QUEUE_WAIT_TIMEOUT=2s`: 队列拥堵 2 秒后快速失败，保护整体吞吐

## 4. 启停命令
启动（后台）:
```bash
cd /root/autodl-tmp/jkh
env RUNPY_MAX_WORKERS=32 RUNPY_MAX_INFLIGHT=128 RUNPY_TIMEOUT=15 RUNPY_QUEUE_WAIT_TIMEOUT=2 RUNPY_WORKER_MAX_TASKS=200 RUNPY_HOST=0.0.0.0 RUNPY_PORT=18080 \
  setsid python tools_server/RunPythonTool.py > tools_server/fastapi.log 2>&1 < /dev/null &
```

查看日志:
```bash
tail -f /root/autodl-tmp/jkh/tools_server/fastapi.log
```

停止:
```bash
pkill -f 'python tools_server/RunPythonTool.py'
```

## 5. 外网访问方式（与本机无关的互联网用户）
当前探测到公网 IP: **205.198.68.59**

外部用户访问 URL:
- 健康检查: `http://205.198.68.59:18080/healthz`
- 接口文档: `http://205.198.68.59:18080/docs`
- 执行接口: `http://205.198.68.59:18080/run`

示例请求:
```bash
curl -X POST 'http://205.198.68.59:18080/run' \
  -H 'Content-Type: application/json' \
  -d '{"raw_text":"```python\nprint(1+1)\nresult=1+1\n```","extract_code":true}'
```

阿里云必须额外确认两点，否则公网无法访问:
1. ECS 安全组放行入方向 TCP `18080`（来源可先 `0.0.0.0/0`，上线建议收敛）
2. 实例系统防火墙放行 `18080`（如有 `ufw/firewalld/iptables`）

## 6. 高并发部署建议（生产）
CPU 密集型任务下，建议每台机器 `gunicorn worker` 不要太多，避免和进程池互相抢核。

推荐（本机 64 vCPU）:
- Gunicorn worker: `1`
- 每 worker 内进程池: `32`

参考命令:
```bash
cd /root/autodl-tmp/jkh
env RUNPY_MAX_WORKERS=32 RUNPY_MAX_INFLIGHT=128 RUNPY_TIMEOUT=15 RUNPY_QUEUE_WAIT_TIMEOUT=2 RUNPY_WORKER_MAX_TASKS=200 \
  gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:18080 tools_server.RunPythonTool:app
```

---

## 7. Search Tool FastAPI（Search-R1 本地检索网关）

### 7.1 实现位置
- 网关代码：`/root/autodl-tmp/jkh/slime/examples/search-r1/local_search_server.py`
- 该文件保留了原始 `local_search(...)` 异步函数（供 Search-R1 训练代码继续调用），并新增了 FastAPI 服务。

### 7.2 网关目标
该网关用于承接外部高并发请求，并转发到本地检索服务（`retrieval_server.py`），核心目标：
- 不阻塞：全异步 I/O（aiohttp + keepalive 连接池）
- 不雪崩：`max_inflight + queue_wait_timeout` 背压机制
- 不无脑重试：有限重试 + 退避
- 可观测：`/stats` 提供吞吐、拥塞、超时统计

### 7.3 接口列表
- `GET /healthz`
- `GET /stats`
- `POST /search`

请求体（`/search`）：
```json
{
  "query": "who invented python",
  "top_k": 3,
  "timeout": 5,
  "proxy": null
}
```

成功返回（HTTP 200）：
```json
{
  "status": "success",
  "query": "who invented python",
  "top_k": 3,
  "count": 3,
  "results": [
    {"document": {"contents": "\"Title\"\nSnippet ..."}}
  ],
  "error": null,
  "latency_ms": 123
}
```

错误返回：
- HTTP `429`：网关排队超时（过载保护）
- HTTP `504`：上游检索超时
- HTTP `502`：上游连接/返回异常

### 7.4 依赖与前置条件
1. 本地检索服务必须可用（默认上游地址 `http://127.0.0.1:8000/retrieve`）。
2. 如果需要公网用户调用，只暴露网关端口（例如 `18081`），上游 `8000` 只允许内网访问。

### 7.5 启动本地检索服务（上游）
> 先启动它，再启动网关。

```bash
python /root/autodl-tmp/jkh/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
  --index_path /your/index/e5_Flat.index \
  --corpus_path /your/corpus/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
```

### 7.6 启动 Search Tool 网关（高并发参数）
本机可用并发核数按 `nproc` 为 **64**，推荐从以下参数起步：
- `LOCAL_SEARCH_API_WORKERS=1`（先避免对上游检索器过度并发轰炸）
- `LOCAL_SEARCH_MAX_INFLIGHT=128`
- `LOCAL_SEARCH_CONNECTOR_LIMIT=512`
- `LOCAL_SEARCH_CONNECTOR_PER_HOST=256`

```bash
cd /root/autodl-tmp/jkh/slime/examples/search-r1
env LOCAL_SEARCH_UPSTREAM_URL=http://127.0.0.1:8000/retrieve \
    LOCAL_SEARCH_API_HOST=0.0.0.0 \
    LOCAL_SEARCH_API_PORT=18081 \
    LOCAL_SEARCH_API_WORKERS=1 \
    LOCAL_SEARCH_MAX_INFLIGHT=128 \
    LOCAL_SEARCH_QUEUE_WAIT_TIMEOUT=1.5 \
    LOCAL_SEARCH_CONNECTOR_LIMIT=512 \
    LOCAL_SEARCH_CONNECTOR_PER_HOST=256 \
    LOCAL_SEARCH_UPSTREAM_TIMEOUT=5 \
    LOCAL_SEARCH_RETRY_TIMES=1 \
    LOCAL_SEARCH_RETRY_BACKOFF=0.2 \
    setsid python local_search_server.py > /root/autodl-tmp/jkh/tools_server/local_search_api.log 2>&1 < /dev/null &
```

当前已启动网关进程（本机）：`PID=334986`

查看日志：
```bash
tail -f /root/autodl-tmp/jkh/tools_server/local_search_api.log
```

停止网关：
```bash
pkill -f 'python local_search_server.py'
```

### 7.7 外网如何访问（与本机无关的任何人）
当前公网 IP：`205.198.68.59`

可访问地址：
- `http://205.198.68.59:18081/healthz`
- `http://205.198.68.59:18081/stats`
- `http://205.198.68.59:18081/search`

阿里云必须配置：
1. ECS 安全组放行 TCP `18081` 入站。
2. 系统防火墙放行 `18081`（`ufw/firewalld/iptables`）。
3. 建议不要对外暴露 `8000`（retrieval 上游端口）。

### 7.8 外部调用示例

`curl`：
```bash
curl -X POST 'http://205.198.68.59:18081/search' \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is python","top_k":3,"timeout":5}'
```

Python 高并发客户端（推荐外部调用方使用连接池 + 超时 + 重试）：
```python
import asyncio
import aiohttp

URL = "http://205.198.68.59:18081/search"
QUERIES = [f"query-{i}" for i in range(500)]

async def one(session, q):
    payload = {"query": q, "top_k": 3, "timeout": 5}
    for _ in range(3):
        try:
            async with session.post(URL, json=payload) as r:
                if r.status in (429, 502, 504):
                    await asyncio.sleep(0.2)
                    continue
                return r.status, await r.json()
        except Exception:
            await asyncio.sleep(0.2)
    return 599, {"status": "failed"}

async def main():
    conn = aiohttp.TCPConnector(limit=300, limit_per_host=300)
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        results = await asyncio.gather(*[one(session, q) for q in QUERIES])
    print("done", len(results))

asyncio.run(main())
```

### 7.9 如何保证“大量并发下吞吐不崩、速度不拖、超时不全挂”
1. 网关层背压：
- `LOCAL_SEARCH_MAX_INFLIGHT` 限制同时在途请求。
- `LOCAL_SEARCH_QUEUE_WAIT_TIMEOUT` 限制排队时间，过载时快速 429，而不是拖到全体超时。

2. 网络层复用：
- 全局 `aiohttp.ClientSession` + `TCPConnector`，减少频繁建连开销。

3. 上游保护：
- 单次请求超时 `LOCAL_SEARCH_UPSTREAM_TIMEOUT`。
- 短退避重试（`LOCAL_SEARCH_RETRY_TIMES` + `LOCAL_SEARCH_RETRY_BACKOFF`）。

4. 可观测性驱动调优：
- 实时看 `/stats`：`busy_requests`、`timeout_requests`、`avg_success_latency_ms`。
- 当 `busy_requests` 快速上升：提高 `max_inflight` 或扩容实例。
- 当 `timeout_requests` 上升：先提升上游检索能力（GPU/索引/批处理），再调高超时。

### 7.10 生产调优建议（按你的 64 vCPU）
初始建议（单机）：
- `LOCAL_SEARCH_API_WORKERS=1`
- `LOCAL_SEARCH_MAX_INFLIGHT=128`
- `LOCAL_SEARCH_CONNECTOR_LIMIT=512`
- `LOCAL_SEARCH_CONNECTOR_PER_HOST=256`
- `LOCAL_SEARCH_UPSTREAM_TIMEOUT=5~8`

逐步调优策略：
1. 如果 429 多、但上游资源空闲：`MAX_INFLIGHT` 每次 +32。
2. 如果 504/502 多：先看上游检索服务是否过载，必要时降 `MAX_INFLIGHT`。
3. 如果 CPU 很低、网关空闲、上游也能扛住：再考虑 `LOCAL_SEARCH_API_WORKERS=2`。

### 7.11 压测建议
可以用 `hey` 或 `wrk` 从外部压测：
```bash
hey -n 5000 -c 200 -m POST \
  -H 'Content-Type: application/json' \
  -d '{"query":"test","top_k":3,"timeout":5}' \
  http://205.198.68.59:18081/search
```

看这三项是否稳定：
- P95/P99 延迟
- 429 比例（过载保护是否生效）
- 502/504 比例（上游是否成为瓶颈）

## 8. 已拉通运行栈（当前生效）

### 8.1 当前在线进程
- `python /root/autodl-tmp/jkh/slime/examples/search-r1/local_dense_retriever/lite_retrieval_server.py`（上游检索，端口 `8000`）
- `python /root/autodl-tmp/jkh/slime/examples/search-r1/local_search_server.py`（Search 网关，端口 `18081`）

### 8.2 为什么使用 lite_retrieval_server
当前机器上未找到官方 dense 索引文件（如 `e5_Flat.index`），因此已启用兼容 `/retrieve` 协议的轻量检索上游：
- 与 Search-R1 `local_search_server.py` 协议完全兼容
- 输入输出结构与原 `retrieval_server.py` 对齐
- 可直接替换为官方检索服务（只要把 `LOCAL_SEARCH_UPSTREAM_URL` 指向原 `/retrieve`）

### 8.3 当前高并发参数
上游 `lite_retrieval_server`：
- `LITE_RETRIEVAL_MAX_INFLIGHT=256`

网关 `local_search_server`：
- `LOCAL_SEARCH_MAX_INFLIGHT=256`
- `LOCAL_SEARCH_CONNECTOR_LIMIT=1024`
- `LOCAL_SEARCH_CONNECTOR_PER_HOST=512`
- `LOCAL_SEARCH_UPSTREAM_TIMEOUT=5`
- `LOCAL_SEARCH_RETRY_TIMES=1`

### 8.4 当前压测结果（已执行）
1) `N=1000, 并发=200`
- 全部成功：`200 x 1000`
- 吞吐：`~1140 req/s`
- 延迟：`p50~119ms, p95~385ms, p99~407ms`

2) `N=10000, 并发=800`
- 全部成功：`200 x 10000`
- 吞吐：`~1944 req/s`
- 延迟：`p50~364ms, p95~520ms, p99~709ms`

网关实时统计（压测后）：
- `peak_inflight_requests=256`
- `busy_requests=0`
- `timeout_requests=0`
- `upstream_error_requests=0`

这说明在当前参数下，面对大量并发 API 调用，系统能维持高吞吐且没有出现排队超时雪崩。

### 8.5 外部访问（任何人）最终说明
外部用户调用地址仍是：
- `POST http://205.198.68.59:18081/search`

若外部无法访问，按顺序检查：
1. 阿里云安全组是否放行 `18081/TCP`。
2. 系统防火墙是否放行 `18081`。
3. 运行环境是否为容器网络（如 `172.17.x.x`），若是，宿主机需做端口映射到容器 `18081`。
4. 不要让请求经过本地代理转发到其他地址（本机调试建议 `curl --noproxy '*' ...`）。

### 8.6 一键重启（上游+网关）
```bash
# 1) 重启上游 lite 检索
pkill -f 'python .*[l]ite_retrieval_server.py' || true
cd /root/autodl-tmp/jkh/slime/examples/search-r1/local_dense_retriever
env LITE_RETRIEVAL_HOST=0.0.0.0 \
    LITE_RETRIEVAL_PORT=8000 \
    LITE_RETRIEVAL_WORKERS=1 \
    LITE_RETRIEVAL_MAX_INFLIGHT=256 \
    LITE_RETRIEVAL_QUEUE_WAIT_TIMEOUT=1.5 \
    setsid python lite_retrieval_server.py > /root/autodl-tmp/jkh/tools_server/lite_retrieval.log 2>&1 < /dev/null &

# 2) 重启 Search 网关
pkill -f 'python .*[l]ocal_search_server.py' || true
cd /root/autodl-tmp/jkh/slime/examples/search-r1
env LOCAL_SEARCH_UPSTREAM_URL=http://127.0.0.1:8000/retrieve \
    LOCAL_SEARCH_API_HOST=0.0.0.0 \
    LOCAL_SEARCH_API_PORT=18081 \
    LOCAL_SEARCH_API_WORKERS=1 \
    LOCAL_SEARCH_MAX_INFLIGHT=256 \
    LOCAL_SEARCH_QUEUE_WAIT_TIMEOUT=1.5 \
    LOCAL_SEARCH_CONNECTOR_LIMIT=1024 \
    LOCAL_SEARCH_CONNECTOR_PER_HOST=512 \
    LOCAL_SEARCH_UPSTREAM_TIMEOUT=5 \
    LOCAL_SEARCH_RETRY_TIMES=1 \
    LOCAL_SEARCH_RETRY_BACKOFF=0.2 \
    setsid python local_search_server.py > /root/autodl-tmp/jkh/tools_server/local_search_api.log 2>&1 < /dev/null &
```
