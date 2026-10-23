# Asphalt AI

Asphalt Legends uchun ekranni ko'rib o'ynaydigan AI. Trassaga emas, ko'rinadigan hodisaga bog'lanadi: ko'rish o'rganiladi,
shuning uchun tayyor AI istalgan trassada o'yinni o'zi baholab o'ynay oladi.

## Tuzilma
| Papka | Vazifa |
|---|---|
| `Core/` | Tezlik o'qigich (OCRsiz): Frame, Glyphs, Reader (raqam + foiz). Platformaga bog'liq emas |
| `Speed/` | Windows konsol: shablon o'rganish/nomlash/o'qish/bench (`MAXG=4` DIST uchun) |
| `App/` | Android APK (`space.asphaltai`): ekran oqimi, virtual gamepad, ko'rish, agent, yozuvchi |
| `Vision/` | PC'da o'rgatish: ko'rish kodlovchisi (CNN), yordamchi yorliqlar, baholash, reward audit |
| `tools/` | Emulator boshqaruvi (`devctl.py`), menyu navigatsiyasi (`flow.py`), o'rgatish tsikli (`cycle.py`) |
| `Depth/` | Tayyor chuqurlik/segmentatsiya tajribalari (hozir faqat yordamchi xususiyat + ONNX eksport) |
| `human data/` | Odam bilimi: harakatlar va burilish katalogi |
| `data/` | Yig'ilgan sessiyalar (`rec_<tag>.bin`, `agent_<tag>.csv`, ...) |

## Arxitektura
1. **APK** o'ynaydi va yozadi: har 250 ms da kichik kadr (96x48), HUD hodisa ustuni, nitro chizig'i, tezlik, DIST%, harakatlar, reward komponentlari -> `rec.bin` + `agent.csv`.
2. **PC** (`Vision/train.py`) kadrlardan **ko'rish kodlovchisini** o'rgatadi. Yorliqlar o'yinning o'zidan (qo'lda belgilash yo'q): to'qnashuv (1 s ichida tezlik yo'qotish), tezlik o'zgarishi, nitro o'sishi, HUD hodisa sinfi (klaster), kelajak reward.
3. Kodlovchi ONNX bo'lib APK assets'iga tushadi (`vision.onnx`, `vision.ver`). Agent (shoxlangan DQN, qurilmada onlayn) kodlovchi chiqargan vektor ustida o'rganadi: **ko'rish yoqiq bo'lsa tayyor segmentatsiya/chuqurlik kerak emas** (`ai eval` ularni o'chiradi).
4. `tools/cycle.py` yig'ish -> o'rgatish -> o'rnatish tsiklini takrorlaydi (ma'lumot to'planadi, kodlovchi yaxshilanadi).

## Reward (App/Agent.cs)
Asos — tezlikni maksimal ushlash. Qo'shimchalar manbalar bilan (GT Sophy, TORCS, Trackmania):
oldinga tezlik (tezlik x yo'l oldinda), devor jarimasi = yo'qotilgan kinetik energiya, yo'ldan chiqish x tezlik, silliq rul, progress (DIST%, o'qigich bor bo'lsa), nitro chizig'i to'lishi.
Tekshiruv: `Vision/reward_audit.py` (komponent hissalari, progress bilan korrelyatsiya, hack alomatlari).

## Buyruqlar (APK, `tools/pad.sh "<buyruq>"`)
`ai start` (o'rganadi), `ai eval` (o'rganmaydi, ko'rish bilan o'ynaydi), `ai watch` (faqat reward hisoblaydi), `ai stop`, `ai reset`; gamepad skriptlari `App/README.md` da.

## Navbat (kod yozilgan, sinov kutilmoqda)
1. `py tools/devctl.py all` — qurish, o'rnatish, yozishni boshlash.
2. DIST shablonlari: `tools/collect_dist.py` -> `tools/dist_templates.py learn/name`.
3. Ma'lumot yig'ish: `py tools/trainloop.py 20 a1` (yoki to'liq tsikl: `py tools/cycle.py`).
4. `cd Vision; py train.py --export; py evaluate.py; py events_sheet.py; py reward_audit.py`.

Bog'liq loyihalar: `V:\Projects\TapperV2` (bot, coach), `V:\Projects\TrackGeo` (jmo geometriya), `V:\Projects\TapperV2\tools\ObjectLab` (ob'ekt qirqish).
