# 全部名义完成模型：最终阶段均值

各训练seed的数值和全部案例保留JSON；不做best-of-checkpoint。不同预算、domain与variant分别列出。

| Track/block/arm | Steps | Variant | Training seeds | Primary | Fresh |
|---|---:|---|---|---:|---:|
| decoder_free/A/b0.1-l0 | 37473 | planning | 1,2 | 4.00% | 4.00% |
| decoder_free/A/b0.1-l0.5 | 37473 | planning | 1,2 | 2.00% | 2.00% |
| decoder_free/A/b0.1-l1 | 37473 | planning | 1,2 | 3.00% | 3.33% |
| decoder_free/A/b0.3-l0 | 37473 | planning | 1,2 | 4.00% | 2.33% |
| decoder_free/A/b0.3-l0.5 | 37473 | planning | 1,2 | 1.33% | 2.67% |
| decoder_free/A/b0.3-l1 | 37473 | planning | 1,2 | 2.67% | 2.00% |
| decoder_free/A/b1-l0 | 37473 | planning | 1,2 | 2.33% | 2.33% |
| decoder_free/A/b1-l0.5 | 37473 | planning | 1,2 | 3.33% | 1.00% |
| decoder_free/B/l0.5-1s | 37473 | planning | 1,2 | 5.67% | 4.67% |
| decoder_free/B/l0.5-om | 37473 | planning | 1,2 | 3.33% | 2.00% |
| decoder_free/B/l0.5-os | 37473 | planning | 1,2 | 2.67% | 2.00% |
| decoder_free/B/l1-1s | 37473 | planning | 1,2 | 4.00% | 2.33% |
| decoder_free/B/l1-os | 37473 | planning | 1,2 | 2.00% | 2.00% |
| decoder_free/C/mean-l0 | 37473 | planning | 1,2 | 3.67% | 2.33% |
| decoder_free/C/mean-l1 | 37473 | planning | 1,2 | 4.00% | 3.33% |
| decoder_free/D/batch | 37473 | planning | 1,2 | 3.33% | 2.67% |
| decoder_free/D/win-l0 | 37473 | planning | 1,2 | 4.00% | 2.00% |
| decoder_free/E/E1 | 37473 | planning | 1 | 4.67% | 4.67% |
| decoder_free/E/E2 | 37473 | planning | 1 | 3.33% | 0.67% |
| decoder_free/E/E3 | 37473 | planning | 1 | 3.33% | 3.33% |
| decoder_free/E/E4 | 37473 | planning | 1 | 6.00% | 5.33% |
| decoder_free/E/E6 | 37473 | planning | 1 | 4.00% | 0.67% |
| decoder_free/E/E7 | 37473 | planning | 1 | 1.33% | 1.33% |
| decoder_free/E/E8 | 37473 | planning | 1 | 0.67% | 0.67% |
| decoder_free/F/F2 | 37473 | planning | 1 | 8.00% | 4.00% |
| decoder_free/F/F3 | 37473 | planning | 1 | 3.33% | 2.00% |
| decoder_free/F/F4-L0 | 45795 | bn | 1 | 1.33% | 1.33% |
| decoder_free/F/F4-L0 | 45795 | raw | 1 | 2.67% | 2.67% |
| decoder_free/G/noinit | 37473 | planning | 1,2 | 1.00% | 1.33% |
| decoder_free/H/s0.3 | 37473 | planning | 1,2 | 4.00% | 2.67% |
| round2/P2/F1 | 24982 | planning | 1,2 | 7.00% | 5.33% |
| round2/P2/F10 | 24982 | planning | 1,2 | 22.00% | 23.00% |
| round2/P2/F2 | 24982 | planning | 1,2 | 74.00% | 69.33% |
| round2/P2/F3 | 24982 | planning | 1,2 | 51.33% | 50.67% |
| round2/P2/F4 | 24982 | planning | 1,2 | 83.00% | 81.67% |
| round2/P2/F5 | 24982 | planning | 1,2 | 7.33% | 8.33% |
| round2/P2/F6 | 24982 | planning | 1,2 | 75.00% | 70.00% |
| round2/P2/F7 | 24982 | planning | 1,2 | 8.00% | 7.33% |
| round2/P2/F8 | 24982 | planning | 1,2 | 87.33% | 83.00% |
| round2/P2/F9 | 24982 | planning | 1,2 | 2.67% | 1.67% |
| round2/P3/a | 24982 | planning | 3,5,6,7 | 5.17% | 5.83% |
| round2/P3/b | 24982 | planning | 3,5 | 4.33% | 6.33% |
| round2/P3/c | 24982 | planning | 3,5 | 4.33% | 3.67% |
| round2/P3/sham | 24982 | planning | 3,5 | 40.33% | 30.33% |
