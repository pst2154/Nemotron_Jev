# Expanded paired endpoint results

Correct denominators include all requested decisions. A failed request contributes no delivered correct answers; it is not a model misclassification. Latencies exclude failed requests. The summary JSON also provides completed-only denominators and accuracy.

| Workload | State tokens | Questions | Backend | Correct / requested | Median ms | p95 ms | Errors |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_lookup | 256 | 1 | lightning_lora | 6/6 | 165 | 179 | 0 |
| random_lookup | 256 | 1 | diffusiongemma | 6/6 | 268 | 281 | 0 |
| random_lookup | 256 | 10 | lightning_lora | 60/60 | 301 | 307 | 0 |
| random_lookup | 256 | 10 | diffusiongemma | 60/60 | 312 | 332 | 0 |
| random_lookup | 1000 | 1 | lightning_lora | 6/6 | 194 | 211 | 0 |
| random_lookup | 1000 | 1 | diffusiongemma | 6/6 | 289 | 306 | 0 |
| random_lookup | 1000 | 10 | lightning_lora | 60/60 | 281 | 286 | 0 |
| random_lookup | 1000 | 10 | diffusiongemma | 60/60 | 359 | 389 | 0 |
| random_lookup | 1000 | 30 | lightning_lora | 177/180 | 335 | 355 | 0 |
| random_lookup | 1000 | 30 | diffusiongemma | 180/180 | 658 | 669 | 0 |
| random_lookup | 1000 | 100 | lightning_lora | 585/600 | 819 | 2,048 | 0 |
| random_lookup | 1000 | 100 | diffusiongemma | 597/600 | 1,806 | 2,008 | 0 |
| random_lookup | 4000 | 1 | lightning_lora | 6/6 | 281 | 311 | 0 |
| random_lookup | 4000 | 1 | diffusiongemma | 6/6 | 549 | 562 | 0 |
| random_lookup | 4000 | 10 | lightning_lora | 60/60 | 624 | 661 | 0 |
| random_lookup | 4000 | 10 | diffusiongemma | 59/60 | 866 | 878 | 0 |
| random_lookup | 4000 | 30 | lightning_lora | 177/180 | 1,462 | 1,596 | 0 |
| random_lookup | 4000 | 30 | diffusiongemma | 180/180 | 1,976 | 2,074 | 0 |
| random_lookup | 4000 | 100 | lightning_lora | 549/600 | 1,011 | 2,162 | 0 |
| random_lookup | 4000 | 100 | diffusiongemma | 599/600 | 6,427 | 6,782 | 0 |
| random_lookup | 8000 | 1 | lightning_lora | 6/6 | 388 | 400 | 0 |
| random_lookup | 8000 | 1 | diffusiongemma | 6/6 | 1,053 | 1,078 | 0 |
| random_lookup | 8000 | 10 | lightning_lora | 60/60 | 596 | 736 | 0 |
| random_lookup | 8000 | 10 | diffusiongemma | 60/60 | 1,860 | 1,877 | 0 |
| random_lookup | 8000 | 30 | lightning_lora | 176/180 | 1,217 | 1,270 | 0 |
| random_lookup | 8000 | 30 | diffusiongemma | 180/180 | 4,494 | 4,526 | 0 |
| random_lookup | 8000 | 100 | lightning_lora | 559/600 | 3,531 | 3,864 | 0 |
| random_lookup | 8000 | 100 | diffusiongemma | 595/600 | 13,947 | 14,202 | 0 |
| random_lookup | 12000 | 1 | lightning_lora | 6/6 | 503 | 531 | 0 |
| random_lookup | 12000 | 1 | diffusiongemma | 6/6 | 1,605 | 1,618 | 0 |
| random_lookup | 12000 | 10 | lightning_lora | 59/60 | 769 | 789 | 0 |
| random_lookup | 12000 | 10 | diffusiongemma | 60/60 | 2,567 | 2,684 | 0 |
| random_lookup | 12000 | 30 | lightning_lora | 172/180 | 1,311 | 1,456 | 0 |
| random_lookup | 12000 | 30 | diffusiongemma | 179/180 | 6,946 | 7,040 | 0 |
| random_lookup | 12000 | 100 | lightning_lora | 540/600 | 3,149 | 3,601 | 0 |
| random_lookup | 12000 | 100 | diffusiongemma | 494/600 | 22,308 | 22,370 | 1 |
| semantic324 | natural | 1 | lightning_lora | 280/324 | 166 | 190 | 0 |
| semantic324 | natural | 1 | diffusiongemma | 244/324 | 252 | 270 | 0 |
| semantic_long | 4000 | 1 | lightning_lora | 16/24 | 237 | 266 | 0 |
| semantic_long | 4000 | 1 | diffusiongemma | 16/24 | 582 | 603 | 0 |
| semantic_long | 12000 | 1 | lightning_lora | 17/24 | 502 | 506 | 0 |
| semantic_long | 12000 | 1 | diffusiongemma | 16/24 | 1,624 | 1,653 | 0 |
| semantic_grouped24 | 1000 | 1 | lightning_lora | 3/6 | 193 | 197 | 0 |
| semantic_grouped24 | 1000 | 1 | diffusiongemma | 6/6 | 305 | 322 | 0 |
| semantic_grouped24 | 1000 | 6 | lightning_lora | 29/36 | 282 | 311 | 0 |
| semantic_grouped24 | 1000 | 6 | diffusiongemma | 36/36 | 331 | 357 | 0 |
| semantic_grouped24 | 1000 | 12 | lightning_lora | 57/72 | 290 | 292 | 0 |
| semantic_grouped24 | 1000 | 12 | diffusiongemma | 71/72 | 473 | 504 | 0 |
| semantic_grouped24 | 1000 | 24 | lightning_lora | 112/144 | 334 | 360 | 0 |
| semantic_grouped24 | 1000 | 24 | diffusiongemma | 144/144 | 683 | 731 | 0 |
| semantic_grouped24 | 4000 | 1 | lightning_lora | 5/6 | 235 | 242 | 0 |
| semantic_grouped24 | 4000 | 1 | diffusiongemma | 6/6 | 564 | 569 | 0 |
| semantic_grouped24 | 4000 | 6 | lightning_lora | 28/36 | 620 | 665 | 0 |
| semantic_grouped24 | 4000 | 6 | diffusiongemma | 33/36 | 592 | 892 | 0 |
| semantic_grouped24 | 4000 | 12 | lightning_lora | 61/72 | 910 | 931 | 0 |
| semantic_grouped24 | 4000 | 12 | diffusiongemma | 70/72 | 922 | 933 | 0 |
| semantic_grouped24 | 4000 | 24 | lightning_lora | 112/144 | 1,404 | 1,481 | 0 |
| semantic_grouped24 | 4000 | 24 | diffusiongemma | 141/144 | 1,880 | 1,886 | 0 |
| semantic_grouped24 | 12000 | 1 | lightning_lora | 4/6 | 553 | 597 | 0 |
| semantic_grouped24 | 12000 | 1 | diffusiongemma | 6/6 | 1,613 | 1,652 | 0 |
| semantic_grouped24 | 12000 | 6 | lightning_lora | 29/36 | 693 | 726 | 0 |
| semantic_grouped24 | 12000 | 6 | diffusiongemma | 34/36 | 1,672 | 2,783 | 0 |
| semantic_grouped24 | 12000 | 12 | lightning_lora | 53/72 | 883 | 920 | 0 |
| semantic_grouped24 | 12000 | 12 | diffusiongemma | 62/72 | 2,748 | 2,793 | 0 |
| semantic_grouped24 | 12000 | 24 | lightning_lora | 111/144 | 1,185 | 1,336 | 0 |
| semantic_grouped24 | 12000 | 24 | diffusiongemma | 130/144 | 5,999 | 6,044 | 0 |

## Semantic 324 by output type

| Backend | Type | Correct |
| --- | --- | ---: |
| lightning_lora | choice | 116/146 |
| lightning_lora | noul | 107/114 |
| lightning_lora | score | 57/64 |
| diffusiongemma | choice | 106/146 |
| diffusiongemma | noul | 109/114 |
| diffusiongemma | score | 29/64 |

## Lookup accuracy by evidence position

| Backend | State tokens | Position | Correct |
| --- | ---: | --- | ---: |
| lightning_lora | 256 | start | 22/22 |
| lightning_lora | 256 | middle | 22/22 |
| lightning_lora | 256 | end | 22/22 |
| lightning_lora | 1000 | start | 281/282 |
| lightning_lora | 1000 | middle | 277/282 |
| lightning_lora | 1000 | end | 270/282 |
| lightning_lora | 4000 | start | 278/282 |
| lightning_lora | 4000 | middle | 266/282 |
| lightning_lora | 4000 | end | 248/282 |
| lightning_lora | 8000 | start | 279/282 |
| lightning_lora | 8000 | middle | 264/282 |
| lightning_lora | 8000 | end | 258/282 |
| lightning_lora | 12000 | start | 280/282 |
| lightning_lora | 12000 | middle | 248/282 |
| lightning_lora | 12000 | end | 249/282 |
| diffusiongemma | 256 | start | 22/22 |
| diffusiongemma | 256 | middle | 22/22 |
| diffusiongemma | 256 | end | 22/22 |
| diffusiongemma | 1000 | start | 281/282 |
| diffusiongemma | 1000 | middle | 282/282 |
| diffusiongemma | 1000 | end | 280/282 |
| diffusiongemma | 4000 | start | 282/282 |
| diffusiongemma | 4000 | middle | 281/282 |
| diffusiongemma | 4000 | end | 281/282 |
| diffusiongemma | 8000 | start | 281/282 |
| diffusiongemma | 8000 | middle | 278/282 |
| diffusiongemma | 8000 | end | 282/282 |
| diffusiongemma | 12000 | start | 181/282 |
| diffusiongemma | 12000 | middle | 276/282 |
| diffusiongemma | 12000 | end | 282/282 |
