# AsphaltApp (space.asphaltai)

.NET 9 Android. Ichida: Core (tezlik o'qigich, `../Core`) + virtual Xbox 360 gamepad (uinput) + MediaProjection ekran oqimi.

```
py tools/devctl.py all         # quradi (Release), o'rnatadi, root sozlaydi, pad yaratadi, yozishni boshlaydi, o'yinni ochadi
```
Emulator (root) uchun deploy.sh qiladi: `appops set space.asphaltai PROJECT_MEDIA allow`, `setenforce 0`, `chmod 666 /dev/uinput`.
(`adb shell su -c '...'` ni FAQAT `adb shell "su -c '...'"` ko'rinishida yozing.)

Pad buyruqlari: `bash tools/pad.sh "<skript>"`. Skript = buyruqlar `;` bilan, ketma-ket bajariladi (`Driver.cs`).

| Buyruq | Ma'nosi |
|---|---|
| `steer -1..1`, `gas 0..1` | chap stik X, RT |
| `drift 1\|0\|tap [ms]`, `nitro 1\|0\|tap [ms]` | ushlab turish / qo'yib yuborish / bosish |
| `shockwave` | nitro 2 marta (gap bilan) |
| `d360` | drift 2 marta tez |
| `punch [gap]` | drift tap, so'ng nitro tap (hujjatdagi "Nitro Punch" talqini) |
| `touchdrive`, `camera`, `respawn`, `menu`, `emote` | D-pad up / D-pad down / B / START / RS |
| `wait ms`, `set tap\|gap ms`, `map drift X` | kutish, vaqtlar, tugmani qayta bog'lash |
| `btn A 1`, `axis X 0.5`, `tap A 80`, `reset`, `stop` | xom / to'xtatish |

Standart xarita (magicgameworld.com, o'yinda TASDIQLANMAGAN): drift=LB, nitro=RB, gaz=RT, rul=chap stik, respawn=B, menu=START, touchdrive=D-pad yuqori, kamera=D-pad past.
Gamepadda alohida tormoz yo'q (manbada ko'rsatilmagan). Barrel roll, Nitro Brake uchun alohida tugma yo'q — `human data` hujjatlarida ham tafsiloti yo'q, skript bilan quriladi.

Tezlik: logcat tegi `AsphaltAI` (`speed=106 digits=106 score=0.98`). Faqat albom ekranda (W >= H); HUD hududi 1560x720 ga nisbatan (1340,20,220x90) masshtablanadi.
"Sinov kadrlar" tugmasi 3 ta poyga kadrini ekranda aylantiradi (o'yinsiz tekshirish).

## Ko'rish (Perception.cs)
SegFormer-B0 (Cityscapes, 352x160) + Depth Anything V2 Small (224x112), ikkalasi ONNX (`Assets/seg.onnx`, `depth.onnx`; `Depth/export_onnx.py` yaratadi), qurilmada Microsoft.ML.OnnxRuntime bilan.
Chiqish: 15 ustun x 5 xususiyat (yo'l bo'sh masofasi, devor, yo'l cheti, mashina, chuqurlik yaqinligi), 0..1. HUD kadrdan olib tashlanadi (pastdagi piksellar bilan to'ldiriladi).
Loglar (Release'da ham): `/sdcard/Android/data/space.asphaltai/files/` — `perc.csv` (t,tezlik,ms,seg_ms,depth_ms,75 xususiyat), `explore.csv`, `dbg.png` (seg + chuqurlik + yo'l ustunlari).
Emulatorda ~0.7 s/xulosa (seg 240 ms + depth 470 ms, Release). PC GPU'da 27 ms.
Skriptlar: `tools/devctl.py`, `tools/pad.sh`, `tools/flow.py` (menyu navigatsiyasi), `tools/trainloop.py`.

## Agent va ko'rish (2026-10-20)
- `Agent.cs`: shoxlangan DQN (rul 5, drift 3, nitro 3, gaz 2), n-qadamli (5) qaytim, gamma 0.97, o'rganish qurilmada, siyosat `policy_<kind>.bin` (kind = `h` qo'lda xususiyatlar yoki `v<versiya>` o'rganilgan kodlovchi).
- `Vision.cs`: `Assets/vision.onnx` bo'lsa yuklanadi (PC'da `Vision/train.py --export`); yo'q bo'lsa agent segmentatsiya+chuqurlik xususiyatlariga qaytadi.
- Yozuvchi: har qadamda `rec.bin` (kadr 96x48, hodisa ustuni, nitro chizig'i) + `agent.csv` (reward komponentlari, DIST, nitro, hodisa faolligi).
- `AiService.Dist`: `Assets/dist_templates.txt` bo'lsa HUD'dan DIST% o'qiladi (yo'q bo'lsa -1, progress reward 0).
- Buyruqlar: `ai start|eval|watch|stop|reset`.
