# Speed — tezlik raqamini tanish (0..605), OCRsiz

Pipeline: oq piksel maskasi (min(R,G,B) >= T) -> bog'langan komponentlar -> uzilgan bo'laklarni birlashtirish -> bir qatordagi katta komponentlar (MPH yozuvi balandligi bilan chiqib ketadi) -> yopishgan raqamlarni ustun minimumi bo'yicha kesish -> 20x24 ga me'yorlash -> shablon NCC (22 shablon, 0-9). T zinasi 225/205/185: eng ko'p raqam + eng yuqori ball tanlanadi, ball < 0.8 bo'lsa rad.

```
Speed learn <crops> <out> [k]      bo'laklarni klasterlash (centers.png)
Speed name  <centers.txt> 1,1,3,... <templates.txt>   klaster -> raqam
Speed read  <templates.txt> <img|dir>
Speed bench <templates.txt> <dir>
```
Shablonlar: `data/templates.txt`. Test kadrlar: `data/crops` (videolardan 120x60 qirqim, 1713 ta).
