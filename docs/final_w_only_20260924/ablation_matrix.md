# 统一六臂成本模型消融（2026-09-27）

每个负载跑同一组臂、同一份 profile、同一资源预算、同一数据量（= 对应 main_fast 的量）。
括号内为相对该负载 `pico_final` 的比例。

| 负载 | PICO（全模型） | Cedar 搜索 + PICO 成本模型 | PICO 搜索 + 旧 Cedar 成本模型 | PICO − 边界项（只改算子层） | PICO − 算子层（只改边界） | Cedar 原生 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simclrv2 | 2348.6（1.00×） | 2375.4（1.01×） | 211.3（0.09×） | 2381.2（1.01×） | 2424.5（1.03×） | 1166.7（0.50×） |
| simclrv2_cache | 2450.7（1.00×） | 2480.8（1.01×） | 215.6（0.09×） | 2457.5（1.00×） | 2458.9（1.00×） | 1182.6（0.48×） |
| commonvoice | 726.8（1.00×） | 727.4（1.00×） | 642.2（0.88×） | 725.3（1.00×） | 725.5（1.00×） | 163.9（0.23×） |
| coco | 217.5（1.00×） | 150.7（0.69×） | 271.3（1.25×） | 214.6（0.99×） | 269.7（1.24×） | 25.4（0.12×） |
| llava_pretrain | — | — | — | — | — | — |
| wikitext103 | — | — | 2447.7 | — | 1511.4 | 5412.8 |

臂身份：

| 臂 | selector | 含义 |
| --- | --- | --- |
| PICO（全模型） | 39 | joint W-only DP search + representation-aware affine compute (k_(op,class)*elements+b) + stage boundary; width fixed at 1 |
| Cedar 搜索 + PICO 成本模型 | 42 | Cedar's staged search priced by the final PICO cost model (Cedar cost function replaced; same compute/boundary/backend/W rules) |
| PICO 搜索 + 旧 Cedar 成本模型 | 28 | DP search priced by Cedar's legacy profile entries (baseline latency + whole-pipeline offload throughput) with Cedar's fixed+bytes boundary; no W search |
| PICO − 边界项（只改算子层） | 40 | same W-only search and element/representation affine compute, explicit boundary term removed (operator-level change only) |
| PICO − 算子层（只改边界） | 41 | byte-proportional compute (y = x in bytes, i.e. Cedar's operator layer) + boundary model + the same W-only search (boundary-level change only) |
| Cedar 原生 | 0 | Cedar native staged optimizer (baseline) |

未产出结果：llava_pretrain